# vision/ocr_async.py
# -*- coding: utf-8 -*-
"""
OCR asynchrone — déporte Tesseract dans un worker thread séparé.

Architecture :
    Boucle principale → met les crops dans ocr_task_queue
    Worker thread     → lit les numéros, met les résultats dans ocr_result_queue
    Boucle principale → fusionne les résultats dans jersey_registry

Le thread principal reste propriétaire de `tracked` et des track_id.
Le worker ne reçoit que des copies de crops (np.ndarray) + metadata.
Aucun objet partagé mutable entre les threads.
"""

import threading
import queue
import time
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from vision.ocr import preprocess_patch, read_number_from_patch, TESSERACT_AVAILABLE


# Tâche envoyée au worker
# (frame_id, track_id, crop_bgr)
OcrTask = Tuple[int, int, np.ndarray]

# Résultat retourné par le worker
# (frame_id, track_id, number_or_None)
OcrResult = Tuple[int, int, Optional[int]]


class AsyncOCRReader:
    """
    Remplace OCRReader avec un worker thread pour Tesseract.

    Usage dans main.py :
        ocr = AsyncOCRReader(min_confidence=0.6, ocr_every_n_frames=7)
        ocr.start()
        ...
        tracked = ocr.enqueue_and_update(frame_orig, tracked, frame_id)
        ...
        ocr.flush()   # attendre toutes les tâches en fin de process_video
        jersey_map = ocr.get_jersey_map()
        ocr.stop()
    """

    def __init__(self, min_confidence: float = 0.6,
                 ocr_every_n_frames: int = 7,
                 max_queue_size: int = 64):
        self.min_confidence      = min_confidence
        self.ocr_every_n_frames  = ocr_every_n_frames
        self._frame_counter      = 0
        self._nom_match          = "test"

        # File de tâches : (frame_id, track_id, crop_bgr)
        self._task_queue: queue.Queue[Optional[OcrTask]] = queue.Queue(maxsize=max_queue_size)

        # Registre des résultats — thread principal seulement
        # {track_id: {"number": int|None, "confidence": float, "votes": dict}}
        self._registry: Dict[int, dict] = {}
        self._registry_lock = threading.Lock()

        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Métriques
        self._n_enqueued  = 0
        self._n_processed = 0
        self._n_dropped   = 0
        self._total_ocr_ms = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────
    def start(self):
        """Démarrer le worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ocr-worker",
            daemon=True
        )
        self._worker_thread.start()

    def stop(self):
        """Arrêter le worker thread proprement."""
        self._running = False
        self._task_queue.put(None)  # signal de fin
        if self._worker_thread:
            self._worker_thread.join(timeout=10)

    def flush(self, timeout: float = 30.0):
        """Attendre que toutes les tâches en file soient traitées."""
        deadline = time.time() + timeout
        while not self._task_queue.empty():
            if time.time() > deadline:
                print(f"  [OCR_ASYNC] flush timeout — {self._task_queue.qsize()} tâches non traitées")
                break
            time.sleep(0.05)
        # Donner un peu de temps au worker pour finir la dernière tâche
        time.sleep(0.1)

    # ── Worker thread ─────────────────────────────────────────────────
    def _worker_loop(self):
        """Boucle du worker — tourne en continu jusqu'à stop()."""
        while True:
            task = self._task_queue.get()
            if task is None:
                break  # signal de fin

            frame_id, track_id, crop = task
            t0 = time.time()

            number = read_number_from_patch(crop) if TESSERACT_AVAILABLE else None

            elapsed_ms = (time.time() - t0) * 1000
            self._total_ocr_ms += elapsed_ms
            self._n_processed += 1

            # Fusionner dans le registre (lock court)
            with self._registry_lock:
                entry = self._registry.get(track_id, {
                    "number": None, "confidence": 0.0, "votes": {}
                })
                if number is not None:
                    votes = entry.get("votes", {})
                    votes[number] = votes.get(number, 0) + 1
                    best = max(votes, key=votes.get)
                    confidence = votes[best] / sum(votes.values())
                    entry = {"number": best, "confidence": confidence, "votes": votes}
                self._registry[track_id] = entry

            self._task_queue.task_done()

    # ── Interface principale ───────────────────────────────────────────
    def enqueue_and_update(self, frame: np.ndarray,
                           players: list, frame_id: int = 0) -> list:
        """
        Remplace ocr.read_all().
        1. Met à jour jersey depuis le registre (cache).
        2. Si frame OCR active : extrait les crops et les envoie au worker.
        3. Ne bloque jamais le thread principal.
        """
        self._frame_counter += 1

        # 1. Mise à jour depuis le registre (lecture seule — lock court)
        with self._registry_lock:
            for p in players:
                tid = p.get("id")
                if tid in self._registry:
                    p["jersey"] = self._registry[tid].get("number")
                else:
                    p["jersey"] = None

        # 2. Frame OCR active ?
        if not TESSERACT_AVAILABLE:
            return players
        if self._frame_counter % self.ocr_every_n_frames != 0:
            return players

        # 3. Extraire crops et envoyer au worker
        h_frame, w_frame = frame.shape[:2]
        for p in players:
            tid = p.get("id")
            if tid is None:
                continue

            # Skip si déjà confiant
            with self._registry_lock:
                entry = self._registry.get(tid, {})
            if entry.get("confidence", 0) >= self.min_confidence:
                continue

            # Extraire crop
            bbox = p.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            pad = int((x2 - x1) * 0.20)
            x1 = max(0, x1 - pad)
            x2 = min(w_frame, x2 + pad)
            y1 = max(0, y1)
            y2 = min(h_frame, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = frame[y1:y2, x1:x2].copy()  # copie — le worker est indépendant
            # Envoyer le crop BGR brut — le worker appellera preprocess_patch lui-même
            try:
                self._task_queue.put_nowait((frame_id, tid, crop))
                self._n_enqueued += 1
            except queue.Full:
                self._n_dropped += 1

        return players

    # ── Jersey map ────────────────────────────────────────────────────
    def get_jersey_map(self) -> dict:
        with self._registry_lock:
            return {
                tid: data["number"]
                for tid, data in self._registry.items()
                if data.get("confidence", 0) >= self.min_confidence
                and data.get("number") is not None
            }

    def reset(self):
        with self._registry_lock:
            self._registry = {}
        self._frame_counter = 0
        self._n_enqueued = 0
        self._n_processed = 0
        self._n_dropped = 0
        self._total_ocr_ms = 0.0

    # ── Stats ─────────────────────────────────────────────────────────
    def print_stats(self):
        avg = self._total_ocr_ms / max(self._n_processed, 1)
        pending = self._task_queue.qsize()
        print(f"  [OCR_ASYNC] enqueued={self._n_enqueued} "
              f"processed={self._n_processed} "
              f"dropped={self._n_dropped} "
              f"pending={pending} "
              f"avg_tesseract={avg:.1f}ms/crop")
