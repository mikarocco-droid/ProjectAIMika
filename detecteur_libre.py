# vision/detecteur_libre.py
# -*- coding: utf-8 -*-
# ════════════════════════════════════════════════════════════════════════════════════════════
# PLAN B LICENCE (15/11/2026) — détecteur sous licence Apache 2.0 pour remplacer Ultralytics YOLO (AGPL-3.0).
#
# RF-DETR (Roboflow) : code ET poids pré-entraînés COCO sous Apache 2.0 pour les tailles Nano, Small, Medium,
# Base et Large (vérifié dans les métadonnées du paquet rfdetr 1.3.0). NE PAS utiliser XL / 2XL (licence PML 1.0,
# hébergement commercial payant). Obligation Apache : livrer le texte de la licence (et un éventuel NOTICE).
#
# Principe : AdaptateurRFDETR imite EXACTEMENT la petite partie de l'interface Ultralytics que le pipeline utilise :
#     resultats = modele(image_ou_liste, conf=…, verbose=False, imgsz=…)
#     for box in resultats[0].boxes:  int(box.cls[0]), float(box.conf[0]), box.xyxy[0].tolist()
# avec la numérotation COCO d'Ultralytics (personne = 0, ballon = 32). Le reste du pipeline (suivi DeepSort,
# ré-identification, ballon HSV…) ne change pas.
#
# ACTIVATION (15/11, fichier SÉPARÉ : detector.py et main.py ne sont PAS modifiés) :
# APRÈS LE RETRAIT D'ULTRALYTICS : RF-DETR est le seul détecteur, toujours actif ; brancher() est gardé pour les
# cellules existantes. Taille : SCOUTIA_RFDETR_TAILLE = medium (défaut) | small | base | large | nano ; poids
# réentraînés : SCOUTIA_RFDETR_POIDS = chemin .pth. Les classes restent numérotées comme avant (personne 0, ballon 32).
# NON VALIDÉ : à tester (test_detecteur_libre.py) puis revalider sur les 10 matchs de référence avant tout usage.
# ════════════════════════════════════════════════════════════════════════════════════════════
import os
import numpy as np

# 15/11 (run Spa sans Ultralytics : suivi 3 à 4 fois plus lent) : en s'important, Ultralytics réglait des paramètres
# globaux du processus, dont cv2.setNumThreads(0) (OpenCV sans multi-fils, pour ne pas concurrencer PyTorch). Sans
# Ultralytics, OpenCV reprend tous ses fils dans CHAQUE processus de suivi (2 en parallèle) → sur-souscription du
# processeur : DeepSort 92 → 470 ms, ré-identification 61 → 201 ms par appel, alors que RF-DETR (GPU) ne ralentissait
# que de 147 à 202 ms. On reproduit ces réglages ici, dès l'import (detector.py importe ce module dans chaque
# processus), quand le détecteur libre est actif. Hypothèse à confirmer par le profilage du prochain run.
# 15/11 (2e run lent malgré cv2) : la vraie cause trouvée dans le code d'Ultralytics 8.4 — son __init__.py fait
#     if not os.environ.get("OMP_NUM_THREADS"): os.environ["OMP_NUM_THREADS"] = "1"
# Importé au démarrage de chaque processus de suivi (versions précédentes), il limitait PyTorch et NumPy à 1 fil de
# calcul par processus. Sans lui, chaque processus (2 en parallèle) prend tous les cœurs → sur-souscription
# (DeepSort 92 → 470 ms/appel). On reproduit les DEUX réglages d'Ultralytics : fils OpenMP/BLAS et cv2.setNumThreads(0).
# La variable d'environnement doit être posée AVANT l'import de torch/numpy : brancher() la pose dans le processus
# principal (les processus de suivi en héritent) ; ici, en secours, torch.set_num_threads(1) si torch est déjà chargé.
def _limiter_fils():
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(min(8, max(1, (os.cpu_count() or 1)))))
    try:
        import cv2 as _cv2_reglage
        _cv2_reglage.setNumThreads(0)
    except Exception:
        pass
    import sys as _sys_fils
    if "torch" in _sys_fils.modules:
        try:
            _sys_fils.modules["torch"].set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
        except Exception:
            pass


if os.environ.get("SCOUTIA_DETECTEUR", "").strip().lower() == "rfdetr":
    _limiter_fils()

# Numérotation COCO « 80 classes » d'Ultralytics, utilisée par detector.py (player_cls = 0, ball_cls = 32)
_IDS_ULTRALYTICS = {"person": 0, "sports ball": 32}
_TAILLES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge"}     # XL / 2XL volontairement absents (PML 1.0)


def detecteur_libre_actif():
    # Ultralytics retiré : RF-DETR est le SEUL détecteur. SCOUTIA_DETECTEUR n'est plus nécessaire (gardé pour compatibilité).
    return True


def _noms_des_classes(modele):
    """Table {numéro RF-DETR: nom COCO}, quelle que soit la version de rfdetr."""
    for source in (lambda: getattr(modele, "class_names", None),
                   lambda: __import__("rfdetr.utilities.coco_classes", fromlist=["COCO_CLASSES"]).COCO_CLASSES,
                   lambda: __import__("rfdetr.utilities", fromlist=["COCO_CLASSES"]).COCO_CLASSES,
                   lambda: __import__("rfdetr.util.coco_classes", fromlist=["COCO_CLASSES"]).COCO_CLASSES):
        try:
            t = source()
        except Exception:
            continue
        if isinstance(t, dict) and t:
            return {int(k): str(v) for k, v in t.items()}
        if isinstance(t, (list, tuple)) and t:
            return {i: str(v) for i, v in enumerate(t)}
    # Dernier recours : numérotation COCO d'origine (personne = 1, ballon = 37), celle de rfdetr ≤ 1.8
    print("  RF-DETR : table des classes introuvable → numérotation COCO d'origine supposée (1 = personne, 37 = ballon)")
    return {1: "person", 37: "sports ball"}


class _Valeur:
    """Imite un tenseur Ultralytics d'une seule ligne : v[0] et v[0].tolist()."""
    def __init__(self, valeur):
        self._v = valeur

    def __getitem__(self, i):
        if i != 0:
            raise IndexError(i)
        return self._v


class _Boite:
    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, cls_id, conf, xyxy):
        self.cls = _Valeur(cls_id)
        self.conf = _Valeur(conf)
        self.xyxy = _Valeur(np.asarray(xyxy, dtype=float))


class _Resultat:
    def __init__(self, boites):
        self.boxes = boites


class AdaptateurRFDETR:
    """Modèle RF-DETR présenté comme un modèle Ultralytics minimal (appel direct, .boxes, classes 0 / 32)."""

    def __init__(self, taille=None, poids=None):
        import rfdetr
        taille = (taille or os.environ.get("SCOUTIA_RFDETR_TAILLE", "medium")).strip().lower()
        if taille not in _TAILLES:
            raise ValueError(f"taille RF-DETR '{taille}' refusée : {sorted(_TAILLES)} (XL / 2XL exclues : licence PML)")
        poids = poids or os.environ.get("SCOUTIA_RFDETR_POIDS", "").strip() or None
        classe = getattr(rfdetr, _TAILLES[taille])
        self.modele = classe(pretrain_weights=poids) if poids else classe()
        # Correspondance des classes PAR NOM (RF-DETR numérote COCO à partir de 1 : personne = 1, ballon = 37).
        # 15/11 : l'emplacement de la table des classes change selon les versions de rfdetr (rfdetr.util supprimé en
        # 1.9.0) → on la cherche à plusieurs endroits, en commençant par celle du modèle lui-même.
        noms = _noms_des_classes(self.modele)
        self._noms = noms
        self._vers_ultralytics = {i: _IDS_ULTRALYTICS[n] for i, n in noms.items() if n in _IDS_ULTRALYTICS}
        # 15/11 : selon la version de rfdetr et le schéma du fichier de poids, la personne sort en 0 OU en 1 (rfdetr
        # 1.8.2 a déplacé « person » de 1 à 0, les anciens poids gardant parfois l'ancien schéma). La table des noms
        # ne suffit donc pas : on CALIBRE sur les premières détections (la classe la plus fréquente d'une image de
        # football est forcément « personne ») — voir _calibrer().
        self._calibre = bool(poids)
        if len(self._vers_ultralytics) != 2:
            raise RuntimeError(f"classes personne / ballon introuvables dans RF-DETR : {self._vers_ultralytics}")
        # Poids réentraînés sur nos images : ce sont NOS classes (à fixer à l'entraînement) — 0 = joueur, 1 = ballon
        if poids:
            _c = os.environ.get("SCOUTIA_RFDETR_CLASSES", "0:0,1:32")
            self._vers_ultralytics = {int(a): int(b) for a, b in (x.split(":") for x in _c.split(","))}
        self.nom = f"rfdetr-{taille}" + (f" ({os.path.basename(poids)})" if poids else " (COCO)")
        self.resolution = getattr(getattr(self.modele, "model", None), "resolution", None)
        self._predicteur = self.modele
        self.optimise = False
        try:                                            # rfdetr ≤ 1.8
            self.modele.optimize_for_inference()
            self.optimise = True
        except Exception:
            try:                                        # rfdetr ≥ 1.9 : « predictor = native_model.inference(...) »
                self._lot_optimise = int(os.environ.get("SCOUTIA_RFDETR_LOT", "4"))   # main.py : lots de 4 images
                _p = self.modele.inference(batch_size=self._lot_optimise)
                if hasattr(_p, "predict"):
                    self._predicteur, self.optimise = _p, True
            except Exception as e:
                print(f"  RF-DETR : optimisation pour l'inférence indisponible ({e}) — vitesse non optimale")
        import sys as _sys_u
        _fils_torch = _sys_u.modules["torch"].get_num_threads() if "torch" in _sys_u.modules else "?"
        print(f"  [processus {os.getpid()}] Ultralytics chargé dans ce processus : "
              f"{'OUI ⚠️' if 'ultralytics' in _sys_u.modules else 'non ✅'} | fils PyTorch : {_fils_torch} "
              f"(attendu : 1) | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
        print(f"  Détecteur LIBRE : {self.nom} | résolution {self.resolution} | optimisé : {'oui' if self.optimise else 'NON'} | licence Apache 2.0 | "
              f"rfdetr {getattr(rfdetr, '__version__', '?')} | classes {self._vers_ultralytics} (RF-DETR → Ultralytics)")

    def __call__(self, source, conf=0.25, verbose=False, imgsz=None, **_ignores):
        # imgsz est ignoré : RF-DETR travaille à sa propre résolution (fixe selon la taille du modèle)
        images = source if isinstance(source, (list, tuple)) else [source]
        rgb = [img[:, :, ::-1].copy() if isinstance(img, np.ndarray) and img.ndim == 3 else img for img in images]  # BGR → RGB
        # 15/11 : le prédicteur optimisé n'accepte qu'une taille de lot fixe (4) ; les autres lots (dernier lot d'un
        # morceau, image seule) passent par le modèle normal, sans désactiver l'optimisation pour la suite.
        _pred = self._predicteur if len(rgb) == getattr(self, "_lot_optimise", len(rgb)) else self.modele
        try:
            dets = _pred.predict(rgb, threshold=float(conf))
        except Exception as e:
            if _pred is self.modele:
                raise
            print(f"  RF-DETR : prédicteur optimisé en échec ({e}) → modèle normal pour la suite", flush=True)
            self._predicteur, self.optimise = self.modele, False
            dets = self.modele.predict(rgb, threshold=float(conf))
        if not isinstance(dets, list):
            dets = [dets]
        if not self._calibre:
            self._calibrer(dets)
        resultats = []
        for d in dets:
            boites = []
            for xyxy, c, k in zip(d.xyxy, d.confidence, d.class_id):
                k_u = self._vers_ultralytics.get(int(k))
                if k_u is not None:
                    boites.append(_Boite(k_u, float(c), xyxy))
            resultats.append(_Resultat(boites))
        self._n_images = getattr(self, "_n_images", 0) + len(resultats)
        self._n_joueurs = getattr(self, "_n_joueurs", 0) + sum(1 for r in resultats for b in r.boxes if b.cls[0] == 0)
        self._n_ballon = getattr(self, "_n_ballon", 0) + sum(1 for r in resultats if any(b.cls[0] == 32 for b in r.boxes))
        if self._n_images in (40, 400) or self._n_images % 4000 < len(resultats):
            print(f"  [RF-DETR processus {os.getpid()}] {self._n_images} images | {self._n_joueurs / self._n_images:.1f} "
                  f"joueurs/image | ballon sur {100 * self._n_ballon / self._n_images:.0f} % des images | "
                  f"optimisé : {'oui' if self.optimise else 'non'}", flush=True)
        return resultats

    def _calibrer(self, dets):
        from collections import Counter
        brut = Counter(int(k) for d in dets for k in d.class_id)
        if not brut:
            return                                     # rien détecté : on réessaiera à l'appel suivant
        # 15/11 (run Kaggle 2) : NE PLUS prendre « la classe la plus fréquente » — sur Minerois Ster, le parking
        # derrière le terrain donne presque autant de voitures (classe 3) que de personnes (1) : un processus de suivi
        # a pu prendre la voiture pour la personne et ne plus voir aucun joueur (2e mi-temps : 0 événement).
        # Règle fixe : la classe 0 n'existe qu'en numérotation « 80 classes » (où 0 = personne, présente sur toute
        # image de football) ; sans 0 → numérotation COCO d'origine (1 = personne, 37 = ballon).
        if 0 in brut:
            personne, ballon = 0, 32
        else:
            personne, ballon = 1, 37
        avant = dict(self._vers_ultralytics)
        self._vers_ultralytics = {personne: 0, ballon: 32}
        self._calibre = True
        print(f"  [processus {os.getpid()}] RF-DETR calibré : classes brutes les plus fréquentes {brut.most_common(5)} "
              f"→ personne = {personne}, ballon = {ballon} (table des noms : {avant})", flush=True)


_ADAPTATEUR = None


def charger_modele(nom):
    """Renvoie le modèle RF-DETR (une seule instance, pour les joueurs ET le ballon), quel que soit le nom demandé
    (« yolo26m.pt » etc. : noms hérités de l'époque Ultralytics, ignorés). Ultralytics a été retiré du projet."""
    global _ADAPTATEUR
    if _ADAPTATEUR is None:
        _ADAPTATEUR = AdaptateurRFDETR()
    return _ADAPTATEUR


def brancher():
    """Conservé pour compatibilité avec les cellules existantes : RF-DETR est désormais toujours utilisé."""
    _limiter_fils()
    print("  Détecteur : RF-DETR (Apache 2.0) — Ultralytics retiré du projet | "
          f"fils de calcul par processus : OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    return True
