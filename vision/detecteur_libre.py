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
# ACTIVATION : variable SCOUTIA_DETECTEUR = "rfdetr" (taille : SCOUTIA_RFDETR_TAILLE = medium | small | base |
# large | nano ; poids réentraînés : SCOUTIA_RFDETR_POIDS = chemin .pth). Non définie = YOLO comme avant.
# NON VALIDÉ : à tester (test_detecteur_libre.py) puis revalider sur les 10 matchs de référence avant tout usage.
# ════════════════════════════════════════════════════════════════════════════════════════════
import os
import numpy as np

# Numérotation COCO « 80 classes » d'Ultralytics, utilisée par detector.py (player_cls = 0, ball_cls = 32)
_IDS_ULTRALYTICS = {"person": 0, "sports ball": 32}
_TAILLES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge"}     # XL / 2XL volontairement absents (PML 1.0)


def detecteur_libre_actif():
    return os.environ.get("SCOUTIA_DETECTEUR", "").strip().lower() == "rfdetr"


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
        self._vers_ultralytics = {i: _IDS_ULTRALYTICS[n] for i, n in noms.items() if n in _IDS_ULTRALYTICS}
        if len(self._vers_ultralytics) != 2:
            raise RuntimeError(f"classes personne / ballon introuvables dans RF-DETR : {self._vers_ultralytics}")
        # Poids réentraînés sur nos images : ce sont NOS classes (à fixer à l'entraînement) — 0 = joueur, 1 = ballon
        if poids:
            _c = os.environ.get("SCOUTIA_RFDETR_CLASSES", "0:0,1:32")
            self._vers_ultralytics = {int(a): int(b) for a, b in (x.split(":") for x in _c.split(","))}
        self.nom = f"rfdetr-{taille}" + (f" ({os.path.basename(poids)})" if poids else " (COCO)")
        self.resolution = getattr(getattr(self.modele, "model", None), "resolution", None)
        try:
            self.modele.optimize_for_inference()
        except Exception as e:
            print(f"  RF-DETR : optimisation pour l'inférence indisponible ({e}) — vitesse non optimale")
        print(f"  Détecteur LIBRE : {self.nom} | résolution {self.resolution} | licence Apache 2.0 | "
              f"rfdetr {getattr(rfdetr, '__version__', '?')} | classes {self._vers_ultralytics} (RF-DETR → Ultralytics)")

    def __call__(self, source, conf=0.25, verbose=False, imgsz=None, **_ignores):
        # imgsz est ignoré : RF-DETR travaille à sa propre résolution (fixe selon la taille du modèle)
        images = source if isinstance(source, (list, tuple)) else [source]
        rgb = [img[:, :, ::-1].copy() if isinstance(img, np.ndarray) and img.ndim == 3 else img for img in images]  # BGR → RGB
        dets = self.modele.predict(rgb, threshold=float(conf))
        if not isinstance(dets, list):
            dets = [dets]
        resultats = []
        for d in dets:
            boites = []
            for xyxy, c, k in zip(d.xyxy, d.confidence, d.class_id):
                k_u = self._vers_ultralytics.get(int(k))
                if k_u is not None:
                    boites.append(_Boite(k_u, float(c), xyxy))
            resultats.append(_Resultat(boites))
        return resultats


_ADAPTATEUR = None


def charger_modele(nom):
    """Remplace `ultralytics.YOLO(nom)` dans detector.py : RF-DETR si SCOUTIA_DETECTEUR=rfdetr (une seule instance,
    pour les joueurs ET le ballon), sinon le modèle Ultralytics demandé, comme avant. Ultralytics n'est importé
    QUE dans ce second cas (aucun import d'Ultralytics quand le détecteur libre est actif)."""
    global _ADAPTATEUR
    if detecteur_libre_actif():
        if _ADAPTATEUR is None:
            _ADAPTATEUR = AdaptateurRFDETR()
        return _ADAPTATEUR
    from ultralytics import YOLO
    return YOLO(nom)
