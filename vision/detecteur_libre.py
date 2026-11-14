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
#     os.environ["SCOUTIA_DETECTEUR"] = "rfdetr"
#     import vision.detecteur_libre as dl ; dl.brancher()      ← AVANT « from pipeline import run_pipeline »
# brancher() remplace, en mémoire seulement, ultralytics.YOLO et vision.detector.YOLO par RF-DETR. Sans cet appel, le
# pipeline tourne exactement comme avant (YOLO). Taille : SCOUTIA_RFDETR_TAILLE = medium (défaut) | small | base |
# large | nano ; poids réentraînés : SCOUTIA_RFDETR_POIDS = chemin .pth.
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
                _p = self.modele.inference()
                if hasattr(_p, "predict"):
                    self._predicteur, self.optimise = _p, True
            except Exception as e:
                print(f"  RF-DETR : optimisation pour l'inférence indisponible ({e}) — vitesse non optimale")
        print(f"  Détecteur LIBRE : {self.nom} | résolution {self.resolution} | optimisé : {'oui' if self.optimise else 'NON'} | licence Apache 2.0 | "
              f"rfdetr {getattr(rfdetr, '__version__', '?')} | classes {self._vers_ultralytics} (RF-DETR → Ultralytics)")

    def __call__(self, source, conf=0.25, verbose=False, imgsz=None, **_ignores):
        # imgsz est ignoré : RF-DETR travaille à sa propre résolution (fixe selon la taille du modèle)
        images = source if isinstance(source, (list, tuple)) else [source]
        rgb = [img[:, :, ::-1].copy() if isinstance(img, np.ndarray) and img.ndim == 3 else img for img in images]  # BGR → RGB
        try:
            dets = self._predicteur.predict(rgb, threshold=float(conf))
        except Exception as e:
            if self._predicteur is self.modele:
                raise
            print(f"  RF-DETR : prédicteur optimisé en échec ({e}) → modèle normal")
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
        return resultats

    def _calibrer(self, dets):
        from collections import Counter
        brut = Counter(int(k) for d in dets for k in d.class_id)
        if not brut:
            return                                     # rien détecté : on réessaiera à l'appel suivant
        personne = brut.most_common(1)[0][0]
        # 15/11 (run Kaggle) : les poids COCO sortent en numérotation COCO D'ORIGINE (91 catégories : personne = 1,
        # ballon = 37), pas en « 80 classes décalées de 1 » (ballon = 33, erreur du premier calibrage). Deux schémas
        # seulement : personne = 0 → 80 classes (ballon = 32) ; personne = 1 → COCO d'origine (ballon = 37).
        if personne == 0:
            ballon = 32
        elif personne == 1:
            ballon = 37
        else:
            print(f"  RF-DETR : classe la plus fréquente inattendue ({personne}) — calibrage abandonné, table des noms gardée")
            self._calibre = True
            return
        avant = dict(self._vers_ultralytics)
        self._vers_ultralytics = {personne: 0, ballon: 32}
        self._calibre = True
        print(f"  RF-DETR calibré sur les détections : classes brutes les plus fréquentes {brut.most_common(5)} "
              f"→ personne = {personne}, ballon = {ballon} (table des noms : {avant})")


_ADAPTATEUR = None
_YOLO_ORIGINAL = None


def charger_modele(nom):
    """Remplace `ultralytics.YOLO(nom)` dans detector.py : RF-DETR si SCOUTIA_DETECTEUR=rfdetr (une seule instance,
    pour les joueurs ET le ballon), sinon le modèle Ultralytics demandé, comme avant. Ultralytics n'est importé
    QUE dans ce second cas (aucun import d'Ultralytics quand le détecteur libre est actif)."""
    global _ADAPTATEUR
    if detecteur_libre_actif():
        if _ADAPTATEUR is None:
            _ADAPTATEUR = AdaptateurRFDETR()
        return _ADAPTATEUR
    if _YOLO_ORIGINAL is not None:
        return _YOLO_ORIGINAL(nom)
    from ultralytics import YOLO
    return YOLO(nom)


def brancher():
    """Branche RF-DETR à la place d'Ultralytics YOLO, EN MÉMOIRE (aucun fichier du pipeline modifié).
    À appeler après os.environ["SCOUTIA_DETECTEUR"] = "rfdetr" et AVANT d'importer / lancer le pipeline.
    detector.py fait « from ultralytics import YOLO » (joueurs) et « from ultralytics import YOLO as _YOLO » (YOLO26m) :
    on remplace donc ultralytics.YOLO et, si le module est déjà importé, vision.detector.YOLO."""
    global _YOLO_ORIGINAL
    if not detecteur_libre_actif():
        print("  Détecteur LIBRE : SCOUTIA_DETECTEUR ≠ rfdetr → rien n'est branché, YOLO reste utilisé")
        return False
    import sys
    import ultralytics
    if _YOLO_ORIGINAL is None:
        _YOLO_ORIGINAL = ultralytics.YOLO
    ultralytics.YOLO = charger_modele
    if "vision.detector" in sys.modules:
        sys.modules["vision.detector"].YOLO = charger_modele
    else:
        import vision.detector as _vd
        _vd.YOLO = charger_modele
    _installer_crochet_sous_processus()
    print("  Détecteur LIBRE branché : ultralytics.YOLO et vision.detector.YOLO → RF-DETR (en mémoire, "
          "et dans les processus de suivi des mi-temps)")
    return True


_CROCHET = '''# sitecustomize.py ÉCRIT PAR vision/detecteur_libre.py (ScoutIA, plan B licence) — ne pas versionner.
# Les mi-temps sont suivies dans des PROCESSUS SÉPARÉS (spawn) : le remplacement en mémoire du processus principal ne
# les atteint pas. Python importe sitecustomize au démarrage de chaque processus dont le PYTHONPATH contient ce
# dossier : on y rebranche RF-DETR, puis on exécute l'éventuel sitecustomize d'origine (Kaggle).
import os, sys
_ici = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("SCOUTIA_DETECTEUR", "").strip().lower() == "rfdetr":
    try:
        import importlib.util, ultralytics
        _spec = importlib.util.spec_from_file_location("detecteur_libre_crochet", os.environ["SCOUTIA_DETECTEUR_FICHIER"])
        _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
        _m._YOLO_ORIGINAL = ultralytics.YOLO
        ultralytics.YOLO = _m.charger_modele
        print(f"  [processus {os.getpid()}] Détecteur LIBRE branché (RF-DETR) dans ce processus de suivi", flush=True)
    except Exception as _e:
        print(f"  [processus {os.getpid()}] ⚠️ Détecteur LIBRE NON branché ({_e}) → YOLO utilisé", flush=True)
for _p in sys.path:
    _f = os.path.join(_p, "sitecustomize.py")
    if _p and os.path.abspath(_p) != _ici and os.path.isfile(_f):
        exec(compile(open(_f, encoding="utf-8").read(), _f, "exec"))
        break
'''


def _installer_crochet_sous_processus():
    """Écrit un sitecustomize.py dans un dossier dédié et l'ajoute au PYTHONPATH : les processus lancés ensuite
    (suivi des mi-temps en parallèle) rebranchent RF-DETR au démarrage."""
    dossier = os.path.join(os.environ.get("TMPDIR", "/tmp"), "scoutia_crochet_detecteur")
    os.makedirs(dossier, exist_ok=True)
    with open(os.path.join(dossier, "sitecustomize.py"), "w", encoding="utf-8") as f:
        f.write(_CROCHET)
    os.environ["SCOUTIA_DETECTEUR_FICHIER"] = os.path.abspath(__file__)
    pp = os.environ.get("PYTHONPATH", "")
    if dossier not in pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = dossier + (os.pathsep + pp if pp else "")
