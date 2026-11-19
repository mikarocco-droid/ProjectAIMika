# ════════════════════════════════════════════════════════════════════════════════════════════
# analysis/montage_joueur_script.py — MONTAGE ET BILAN D'UN JOUEUR (version de production, 19/11/2026)
# NE PAS LANCER DIRECTEMENT : exécuté par analysis/montage_joueur.py (fonction monter_joueur), qui fournit les réglages
# dans PARAMS_MONTAGE. Contenu = la cellule Kaggle validée (n° 6 de SRU Verviers, gardien n° 1) — lecture des numéros
# PARSeq v2, mode gardien (couleur + vérification Gemini, arrêts, sans les 6 mètres), tous les buts du match, fiche.
# Les chemins Kaggle sont remplacés par : ENTREES (dossiers où chercher lectures, réponses Gemini, poids PARSeq) et
# SORTIE (dossier des livrables). Les réglages Kaggle d'origine restent des valeurs par défaut, remplacées par PARAMS.
# ════════════════════════════════════════════════════════════════════════════════════════════
import os
import glob as _glob_p
P = dict(globals().get("PARAMS_MONTAGE") or {})
ENTREES = list(P.get("ENTREES") or ["/kaggle/input"])
SORTIE = P.get("SORTIE") or "/kaggle/working"
os.makedirs(SORTIE, exist_ok=True)


import collections as _c_p
COUT_ANALYSE = _c_p.defaultdict(lambda: [0, 0.0])      # coût de l'analyse du match : compté par le pipeline, pas ici
DETECTEUR_TAILLE = P.get("DETECTEUR_TAILLE") or os.environ.get("SCOUTIA_RFDETR_TAILLE", "medium")
DUREE_ANALYSE_MIN = float(P.get("DUREE_ANALYSE_MIN") or 0.0)
_att = P.get("ATTRIBUTION")


def _ecrire_resultat(**d):
    """Résultat du montage pour monter_joueur() (JSON dans SORTIE)."""
    import json as _jr
    d.setdefault("match", globals().get("MATCH")); d.setdefault("numero", globals().get("NUMERO"))
    _jr.dump(d, open(os.path.join(SORTIE, "resultat_montage_joueur.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _cherche(motif):
    """Fichiers correspondant au motif dans les dossiers ENTREES (récursivement)."""
    return sorted({f for d in ENTREES for f in _glob_p.glob(os.path.join(d, "**", motif), recursive=True)})


# ════ ÉTAPE 2 — MONTAGE DU JOUEUR (kaggle_joueur_v3_2.py) ════
# ════════════════════════════════════════════════════════════════════════════════════════════
# ScoutIA — MONTAGE JOUEUR v3.2 : PARSeq (v2) + CONFIRMATION GEMINI des segments lus une seule fois (07/11/2026)
# Constat du test n°16 : le modèle repère souvent le bon numéro UNE fois dans un segment où le joueur est bien là
# (règle : ≥ 2 lectures) — surtout sur les maillots pâles à numéros foncés. Ces segments, et seulement eux, sont relus
# par Gemini (4 vignettes, ≈ 0,007 $ chacun) ; la règle de retenue (≥ 2 lectures, aucun autre numéro) est inchangée.
# ─ base : test n°16 ─────────────────────────────────────────────────────────────────────────
# ScoutIA — TEST n°16 bleu ciel : modèle v1 contre v2 sur le résumé de Minerois Ster (07/11/2026)
# Réglages du test : NUMERO 16, COULEUR « bleu clair », FENETRES = les 21 actions du résumé, MONTAGE « presence »,
# PARSEQ_VERSION à mettre à "v1" puis à "v2". Vérité : participations à 81:51, 87:16, 89:53, 110:08.
# ─ base : v3.1 ─────────────────────────────────────────────────────────────────────────────
# ScoutIA — MONTAGE D'UN JOUEUR, v3.1 : TOUT LE MATCH + TOUCHES DE BALLE du joueur (06/11/2026)
# Nouveau v3.1 : FENETRES vide = tout le match est lu (gratuit avec PARSeq) ; les événements du suivi (possession,
# passe, dribble, tir…) attribués à une piste du joueur PENDANT une portion où son numéro est lu sont listés comme
# « touches », et MONTAGE = "touches" monte un extrait autour de chacune (sinon "presence" = v2.8).
# Réserve : ces événements viennent du suivi du ballon, peu fiable avec cette caméra — À VÉRIFIER dans la vidéo.
# ─ v3.0 ─────────────────────────────────────────────────────────────────────────────────────
# ScoutIA — MONTAGE D'UN JOUEUR, v3.0 : résumé + repère, numéros lus par NOTRE modèle PARSeq affiné (05/11/2026)
# Nouveau v3.0 : la lecture des numéros n'utilise plus Gemini mais parseq_scoutia_maillots_v1.pt (PARSeq © Darwin
# Bautista, Apache 2.0, affiné sur les lectures Gemini de Franchimont). Gratuit, donc AUCUN plafond. Seuil 0,8.
# Les couleurs des équipes viennent toujours de Gemini (2 questions). Lectures dans lectures_v30_<match>.json
# (jamais mélangées avec celles de Gemini). Tout le reste (segments, règle de retenue, raccords, repère) = v2.9.
# ─ v2.9 ─────────────────────────────────────────────────────────────────────────────────────
# Nouveau v2.9 : FENETRES = heures des actions du résumé. Seuls les segments du suivi qui touchent ces fenêtres sont lus
# par Gemini (coût réduit), et le montage contient TOUTES les fenêtres (le résumé complet), avec le repère dessiné
# là où le joueur est reconnu. Reste identique à la v2.8 (même lecture, mêmes raccords, même repère).
# ─ v2.8 ─────────────────────────────────────────────────────────────────────────────────────
# Ne modifie NI le suivi NI la détection des buts. CPU ; Gemini : ~1 question par segment lu (plafonné).
#
# Base : v2.7 (lecture en réflexion limitée, contrôle des buts). Avec CONSENSUS_TEMPORAL = False (défaut), le
# montage est construit EXACTEMENT comme en v2.7 (seuls changent les noms de fichiers et une colonne « consensus »
# à 0 dans les CSV).
#
# Nouveau v2.8 (option CONSENSUS_TEMPORAL = True) : CONSENSUS DANS LE TEMPS, testé le 29/10 sur Franchimont
# (+79 s de présence du 10, dégagement raté de 49:37 retrouvé, 15 groupes vérifiés à l'œil).
#  - Une lecture « 10 » isolée est acceptée si d'autres lectures « 10 » existent à moins de CONS_S s, au même
#    endroit de l'image ; le groupe doit avoir ≥ CONS_MIN_LECTURES lectures dont ≥ CONS_MIN_ROUGES sur des
#    vignettes ROUGES ; aucune lecture sur une vignette de l'équipe adverse (elle a aussi un n° 10).
#  - Lectures simultanées à deux endroits (≤ 0,2 s, > 0,5 hauteur) : seule la lecture la MOINS soutenue par le
#    reste du groupe est exclue (les deux si égalité), puis le groupe est revalidé.
#  - Groupes faibles (pas assez de lectures rouges après exclusion) : jamais ajoutés, mais tracés.
#  - Sources : lectures normales + lectures renforcées (lectures_renforcees_<match>.json en entrée, lecture seule).
#  - Présence : ± GARDE_S autour de chaque lecture gardée, sans franchir une vignette de l'équipe adverse ou d'un
#    autre numéro ; ajoutée seulement là où le joueur n'est pas déjà repéré (pas de doublon).
#  - Montage : repère ORANGE « #10* » pour le consensus (jaune = lu, bleu = raccord). Trace :
#    consensus_v28_*.csv et verification_consensus_v30_p*.jpg.
# Validé seulement sur le cache en mode « filtre » : à reconfirmer sur le run complet en mode « historique ».
#
# MODES : "diagnostic" (cache seul, gratuit) | "essai" (pistes TIDS_DIAG seulement ; raccords cherchés dans tout
#         le cache) | "complet".
# ════════════════════════════════════════════════════════════════════════════════════════════
# ── RÉGLAGES ──────────────────────────────────────────────────────────────────────────────────
MODE         = "complet"      # "diagnostic", "essai" ou "complet"
MATCH        = "sruverviers"
VIDEO        = "/kaggle/input/datasets/michaelziant/sruverviers/p4-sru-verviers-b-2026-10-10.mp4"
CACHE        = ""
                              # dossier ou archive du cache du suivi ; vide = recherche automatique
NUMERO       = 1
COULEUR      = "bleu"   # COULEUR DE L'ÉQUIPE (joueurs de champ) ; rouge, bleu, bleu clair, blanc, noir, jaune, vert, orange, violet, gris
PARSEQ_VERSION = "v2"     # ← TEST n°16 : lancer une fois avec "v1", une fois avec "v2" ("" = le plus récent disponible)
GARDIEN         = False    # 19/11 : mode gardien (reconnu par la couleur de SON maillot et sa position)
COULEUR_MAILLOT = ""       # couleur du maillot du gardien (différente de celle de son équipe)
HAUTEUR_MIN_GARDIEN = 40   # 19/11 (5) : px — pas de numéro à lire, le gardien loin de la caméra doit compter
IMAGES_PAR_PASSAGE_GARDIEN = 3   # 19/11 (7) : 3 images suffisent pour dire « noir » ou « pas noir »
MAX_VERIF_GARDIEN   = 150  # personnes-périodes vérifiées au plus (19/11 (6) : regroupées par piste du suivi)
VIGNETTES_PAR_QUESTION = 6 # personnes par question Gemini (planche de 6 images)
EQUIPE       = None           # None = équipe choisie par la couleur ; 0 ou 1 = équipe imposée
MARGE_AVANT  = 4.0            # secondes de contexte avant une portion retenue
MARGE_APRES  = 3.0            # secondes après
FUSION_S     = 8.0            # deux portions à moins de 8 s = même séquence
MIN_DUREE_S  = 2.0            # segment lu s'il dure au moins 2 s
MIN_HAUTEUR  = 90             # hauteur minimale du joueur (px) pour qu'un numéro soit lisible
N_LECTURES   = 12             # 09/11 : 12 avec la LECTURE RAPIDE (le temps ne dépend presque plus du nombre d images : la vidéo est décodée une fois)
MIN_CONCORDANTES = 2          # le numéro doit être lu au moins 2 fois sur des vignettes de la bonne couleur
LECTURE_UNIQUE_SI_COULEUR = False   # True = accepter UNE lecture si ≥ 2 vignettes consécutives de la bonne
                                    # couleur et aucun autre numéro lu (hypothèse à tester, pas validée)
MAX_SEGMENTS = 0              # segments EXAMINÉS en mode complet (couleur mesurée, sans Gemini) ; 0 = tous
MAX_QUESTIONS = 10**9           # plafond de NOUVELLES questions Gemini par run (lectures réutilisées non comptées)
TENUE_REPERE_S = 0.5          # repère tenu au plus 0,5 s après la dernière position réelle
# v2.9 : résumé du match (heures des actions repérées par le pipeline) ; [] = comportement v2.8 (tout le match)
FENETRES = ["15:21", "18:27", "23:28", "25:12", "30:23", "42:48", "50:15", "55:42", "74:51", "81:51", "87:16",
            "89:53", "90:57", "92:38", "96:29", "99:02", "103:02", "106:53", "110:08", "111:35", "119:50"]   # test : le résumé
MONTAGE = "touches" if True else "presence"  # v3.1 : "touches" = un extrait autour de chaque touche de balle ; "presence" = comportement v2.8
TOUCHE_AVANT, TOUCHE_APRES = 4.0, 3.0
TYPES_TOUCHES = {"possession", "pass", "long_pass", "dribble", "shot", "interception", "progressive_run", "fast_break", "goal"}
FENETRE_AVANT, FENETRE_APRES = 10.0, 15.0
# ── PRODUCTION : les réglages fournis par monter_joueur() remplacent les valeurs ci-dessus ──
for _k_p, _v_p in P.items():
    if _k_p.isupper() and _k_p not in ("ENTREES", "SORTIE", "RELAIS", "GEMINI_KEY", "PARSEQ_POIDS"):
        globals()[_k_p] = _v_p
# ── BOUT EN BOUT (07/11) : si la cellule 1/2 a tourné dans ce notebook, ses résultats remplacent les réglages ──
import os, glob as _gbe, json as _jbe, time as _tbe
T_DEBUT_MONTAGE = _tbe.time()
# 18/11 : FICHE « BILAN DU JOUEUR » (module export/fiche_joueur.py du dépôt) jointe au rapport du match s'il est présent
def _fiche_joueur(trouve, presence_s=0.0, n_passages=0, n_moments=0, buts_conf=(), buts_poss=(), **opt_fiche):
    try:
        import subprocess as _sp_f, sys as _sys_f, glob as _g_f
        try:
            import export.fiche_joueur as _fj_test  # noqa: F401  (production : le dépôt est déjà importable)
        except ImportError:
            pass
        if "export.fiche_joueur" not in _sys_f.modules and not os.path.isdir("/kaggle/working/ProjectAIMika"):
            _sp_f.run(["git", "clone", "-q", "--depth", "1", "https://github.com/mikarocco-droid/ProjectAIMika.git",
                       "/kaggle/working/ProjectAIMika"], check=False)
        if "export.fiche_joueur" not in _sys_f.modules:
            _sp_f.run([_sys_f.executable, "-m", "pip", "install", "-q", "fpdf2", "pypdf"], check=False)
        if "export.fiche_joueur" not in _sys_f.modules and "/kaggle/working/ProjectAIMika" not in _sys_f.path:
            _sys_f.path.insert(0, "/kaggle/working/ProjectAIMika")
        from export.fiche_joueur import generer_fiche_joueur, joindre_au_rapport
        from export.rapport_client import minute_de_match
        _bn = (BOUT_EN_BOUT or {}).get("bornes") or {}
        if isinstance(_bn, (list, tuple)):      # relais : [[KO1, fin 1re MT], [KO2, fin du match]]
            _bn = ({"ko1": float(_bn[0][0]), "fin1mt": float(_bn[0][1]), "ko2": float(_bn[1][0]), "finmatch": float(_bn[1][1])}
                   if len(_bn) >= 2 else {"ko1": float(_bn[0][0])} if _bn else {})
        _ko1 = float(_bn.get("ko1") or KO)
        _mn = lambda g: minute_de_match(g, _ko1, _bn)
        _raison = lambda r: (r or "").replace("touche(s) dans les 15 s avant", "touche(s) juste avant") \
                                     .replace("reconnu ≈", "présent").replace("dans les 8 s avant", "juste avant")
        _fp = generer_fiche_joueur(f"{SORTIE}/fiche_joueur_{MATCH}_{NUMERO}.pdf", NUMERO,
                                   (f"gardien, équipe {COULEUR}" if GARDIEN else COULEUR), trouve,
                                   presence_s=presence_s, n_passages=n_passages, n_moments_ballon=n_moments,
                                   buts_confirmes=[(_mn(g), _raison(r) or "présent") for g, r in buts_conf],
                                   buts_possibles=[(_mn(g), _raison(r)) for g, r in buts_poss], match=MATCH,
                                   **opt_fiche)
        print(f"   FICHE JOUEUR : {_fp}")
        _rap = ([P["RAPPORT_PDF"]] if P.get("RAPPORT_PDF") and os.path.exists(P["RAPPORT_PDF"]) else []) or \
            sorted(_g_f.glob(f"{SORTIE}/rapport_{MATCH}*.pdf") + _g_f.glob(f"{SORTIE}/apercu_{MATCH}.pdf")
                   + _cherche(f"rapport_{MATCH}*.pdf"))
        _rap = sorted([r for r in _rap if "_joueur" not in r],
                      key=lambda r: (r == P.get("RAPPORT_PDF"), "optionB" in r, r.startswith(SORTIE), r))   # option B et run courant d'abord
        if _rap:
            _out = joindre_au_rapport(_rap[-1], _fp, f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf")
            if _out:
                print(f"   RAPPORT + FICHE : {_out} (rapport utilisé : {_rap[-1]})")
        else:
            print("   (aucun rapport du match trouvé : la fiche reste seule)")
    except Exception as _e_f:
        print(f"   ⚠️ fiche du joueur non produite : {_e_f}")


# 18/11 : relais du match demandé UNIQUEMENT (le 1er run a pris par erreur bout_en_bout_minerois_ster.json de codeprojectai)
_RELAIS_P = P.get("RELAIS")          # production : relais fourni par le pipeline (match, vidéo, cache, bornes, buts)
_be = None if _RELAIS_P else next(iter(sorted(_gbe.glob(f"{SORTIE}/bout_en_bout_{MATCH}.json"))
                                       + _cherche(f"bout_en_bout_{MATCH}.json")), None)
if not _be and not _RELAIS_P:
    raise SystemExit(f"❌ bout_en_bout_{MATCH}.json introuvable — rien n'est lancé")
BOUT_EN_BOUT = None
# 08/11 : relais de l'étape 1 COLLÉ ICI (utilisé si aucun fichier bout_en_bout_*.json n'est trouvé). Mettre None pour l'ignorer.
RELAIS_MANUEL_ANCIEN = {"match": "minerois_ster", "video": "/kaggle/input/datasets/michaelziant/mineroisster-complet/minerois-ster-2025-11-09.mp4",
                 "cache": "/kaggle/working/ProjectAIMika/outputs/minerois_ster_reference/cache", "actions_abs": [],
                 "bornes": [[723.0, 3445.0], [4271.0, 7227.0]],
                 "buts": [{"time_abs": 3341.0, "statut": "?"}, {"time_abs": 5236.0, "statut": "?"}, {"time_abs": 5961.0, "statut": "?"},
                          {"time_abs": 6183.0, "statut": "?"}, {"time_abs": 6612.0, "statut": "?"}, {"time_abs": 7190.0, "statut": "?"}],
                 "t0": None, "duree_analyse_min": 100.9, "joueur_seul": True}
RELAIS_MANUEL = _RELAIS_P   # production : relais fourni par monter_joueur()
if _be or RELAIS_MANUEL:
    BOUT_EN_BOUT = _jbe.load(open(_be, encoding="utf-8")) if _be else dict(RELAIS_MANUEL)
    if BOUT_EN_BOUT.get("match") != MATCH:
        raise SystemExit(f"❌ le relais est celui de {BOUT_EN_BOUT.get('match')}, pas de {MATCH} — rien n'est lancé")
    print(f"   relais : {'fichier ' + _be if _be else ('fourni par le pipeline' if _RELAIS_P else 'RELAIS_MANUEL (collé dans le script)')}")
    MATCH, VIDEO, CACHE = BOUT_EN_BOUT["match"], BOUT_EN_BOUT["video"], BOUT_EN_BOUT["cache"]
    if not os.path.exists(os.path.join(CACHE, "frames_data.pkl")):
        # 08/11 : cellule 2 lancée dans un AUTRE notebook (sortie de la cellule 1 en entrée) : on décompresse le cache neuf
        import zipfile as _zbe
        # ordre : dossier « cache_<match>_neuf » (dataset déjà décompressé), puis zip « _neuf » ou « _reference » de la
        # SORTIE de l'étape 1. JAMAIS l'ancien dossier codeprojectai/cache_<match>_reference (autre suivi).
        _tous = sorted(_cherche("frames_data.pkl"))
        print("   caches de suivi présents dans les entrées : " + (" | ".join(_tous) if _tous else "aucun"))
        # Kaggle peut décompresser le zip de la sortie de l'étape 1 en un dossier « cache_<match>_reference » : on l'accepte
        # S'IL N'EST PAS dans codeprojectai (là se trouve l'ancien suivi, du même nom).
        def _date_cache(f):
            try:
                m = _jbe.load(open(os.path.join(os.path.dirname(f), "meta.json"), encoding="utf-8"))
                return str(m.get("saved_at") or m.get("date") or "")
            except Exception:
                return ""
        _t0_relais = BOUT_EN_BOUT.get("t0")
        _jour_relais = _tbe.strftime("%Y-%m-%d", _tbe.gmtime(_t0_relais)) if _t0_relais else "2026-11-07"
        for f in _tous:
            if f"cache_{MATCH}" in f:
                print(f"     {f} : sauvegardé le {_date_cache(f) or '?'} (étape 1 du {_jour_relais})")
        # nouveau cache = nommé « _neuf », OU hors codeprojectai, OU (même dans codeprojectai) sauvegardé le jour de l'étape 1
        # ou après : l'utilisateur a pu remplacer l'ancien cache par le nouveau sous le même nom.
        _dos = next(iter([f for f in _tous if f"cache_{MATCH}_neuf" in f]
                         + [f for f in _tous if f"cache_{MATCH}_reference" in f and "/codeprojectai/" not in f]
                         + [f for f in _tous if f"cache_{MATCH}_reference" in f and _date_cache(f)[:10] >= _jour_relais]), None)
        _z = next(iter(sorted(_cherche(f"cache_{MATCH}_neuf.zip"))
                       + sorted(_cherche(f"cache_{MATCH}_reference.zip"))), None)
        if _dos:
            CACHE = os.path.dirname(_dos)
        elif _z:
            _dest = f"{SORTIE}/cache_{MATCH}_neuf"
            if not os.path.exists(os.path.join(_dest, "frames_data.pkl")):
                _zbe.ZipFile(_z).extractall(_dest)
            _fd = next(iter(_gbe.glob(f"{_dest}/**/frames_data.pkl", recursive=True)), None)
            if not _fd:
                raise SystemExit(f"❌ frames_data.pkl absent de {_z}")
            CACHE = os.path.dirname(_fd)
        else:
            raise SystemExit(f"❌ cache NEUF introuvable : ajouter en entrée la sortie de l'étape 1 (cache_{MATCH}_reference.zip), "
                             f"ou un dataset « cache_{MATCH}_neuf » — l'ancien cache de codeprojectai n'est volontairement pas utilisé")
        print(f"   cache NEUF utilisé : {CACHE}")
    FENETRES = [f"{int(t) // 60:02d}:{int(t) % 60:02d}" for t in BOUT_EN_BOUT["actions_abs"]]
    print(f"   BOUT EN BOUT : {MATCH} | cache neuf {CACHE} | {len(FENETRES)} action(s) du résumé produites par le pipeline")
else:
    print("   ⚠️⚠️ AUCUN RELAIS bout_en_bout_*.json TROUVÉ (ni dans /kaggle/working, ni dans les entrées) : réglages PAR DÉFAUT "
          "(ancien cache de codeprojectai, résumé de 21 actions). Pour le test « joueur seul », ajouter en entrée la SORTIE "
          "du notebook de la cellule 1 (Add Input → Notebook Output Files).")
# découpage des pistes (hypothèses à valider)
COUPURE_TROU_S   = 1.0        # trou plus long → nouveau segment
SAUT_BASE_H      = 1.0        # déplacement toléré entre deux positions : SAUT_BASE_H + SAUT_H_PAR_S × durée,
SAUT_H_PAR_S     = 6.0        #   en hauteurs de joueur (≈ 6 hauteurs/s = sprint)
RAPPORT_TAILLE_MAX = 1.6      # changement de hauteur du cadre plus brusque → nouveau segment
EXIGER_TSU   = False          # True = arrêter si le cache n'a pas time_since_update
PREFILTRE_EQUIPE = 0.0        # 07/11 : 0 (était 0,1). Le cache étiquette mal l équipe bleu ciel (souvent « None ») : des morceaux du n°16 de 150 à 245 px (de près !) étaient ignorés. La couleur mesurée vignette par vignette protège toujours l identité.
                              # (champ peu fiable : simple tri pour limiter le coût ; 0 = pas de tri)
SEUIL_COULEUR = 0.15          # part minimale des pixels du haut du maillot dans la couleur de l'équipe
TIDS_DIAG    = ["2383", "6824", "7969"]   # pistes examinées (diagnostic) ou lues (essai)
FICHIER_LECTURES = ""         # vide = /kaggle/working/lectures_v30_<match>.json ; fusionne v2.8 à v2.2
# raccordement (hypothèses à valider)
RACCORD_MAX_S    = 2.0        # écart maximal entre la fin d'une portion et le début du segment voisin
RACCORD_RECOUV_S = 1.0        # chevauchement toléré (deux pistes sur le même joueur au même moment)
RACCORD_BASE_H   = 1.0        # distance tolérée au raccord : RACCORD_BASE_H + RACCORD_H_PAR_S × écart,
RACCORD_H_PAR_S  = 3.0        #   en hauteurs de joueur (la caméra Veo bouge : distance dans l'image, pas sur le terrain)
RACCORD_MAX_CHAINE = 4        # raccords successifs au plus, dans chaque sens
RACCORD_HMIN     = 40         # hauteur minimale (px) d'une vignette de couleur pour un raccord
MARQUER_RACCORDS = True       # repère bleu « #10? » sur les raccords (vérification) ; False = jaune partout
# variantes de raccord (v2.5) : calculées et comparées ; le montage utilise RACCORD_MONTAGE
RACCORD_RAPPORT  = 2.0        # B : le plus proche est accepté si le suivant est au moins 2 fois plus loin
RACCORD_RECOUV_MAX_H = 0.3    # v2.6 : pistes qui se chevauchent dans le temps → même joueur seulement si ≤ 0,3 hauteur
EXTENSION_MAX_S  = 2.0        # v2.6 : prolongement maximal d'une portion lue au-delà des vignettes lues
RACCORD_MONTAGE  = "B"        # "ref" ou "B" (« B partiel (v2.4) » : comparaison seulement, jamais monté)
LIGNES_PAR_PAGE  = 6          # 07/11 : 6 (était 20). Avec 12 vignettes par ligne, les pages plus hautes étaient réduites à l envoi
MODELE       = "gemini-3.8-flash"
REFLEXION_LECTURE = "low"     # v2.7 : réflexion limitée pour la lecture des numéros ; None = réflexion normale
CONTROLE_BUT_AVANT_S = 20     # v2.7 : contrôle des buts, fenêtre avant le but (s)
# v2.8 : consensus dans le temps (désactivé par défaut : comportement v2.7 inchangé)
CONSENSUS_TEMPORAL = (not GARDIEN)     # 19/11 : pas en mode gardien (repose sur les numéros) ; v2.9 : activé pour étendre le repère (sans nouvelle question Gemini)
CONS_S, CONS_BASE_H, CONS_H_PAR_S = 2.0, 1.0, 3.0   # lectures liées : ≤ 2 s, distance ≤ 1 + 3 × écart (hauteurs)
CONS_MIN_LECTURES, CONS_MIN_ROUGES = 2, 2           # groupe valable : ≥ 2 lectures « 10 », dont ≥ 2 sur vignettes rouges
GARDE_S = 1.0                                       # présence gardée ± 1 s autour de chaque lecture
# coût (29/10) : tarif de la passation pour gemini-3.8-flash, en dollars par million de jetons. Les jetons de
# RÉFLEXION sont comptés au prix de la sortie (règle de facturation Google à revérifier sur la page des tarifs).
PRIX_ENTREE_M = 0.75
PRIX_SORTIE_M = 3.75
TAUX_EUR      = 0.92          # conversion approximative, pour l'affichage seulement

import os, glob, json, pickle, zipfile, subprocess, collections, csv, time
import numpy as np
import cv2

T0 = time.time()
SORTIE = SORTIE   # production : fixé dans le préambule (PARAMS_MONTAGE["SORTIE"])
fmt = lambda t: f"{int(t)//60:02d}:{int(t)%60:02d}"
dans_image = lambda b: 0 <= b[0] < b[2] <= 1920 and 0 <= b[1] < b[3] <= 1080

if MODE not in ("diagnostic", "essai", "complet"):
    raise SystemExit(f"❌ MODE = {MODE!r} : choisis « diagnostic », « essai » ou « complet »")


def _cle_gemini():
    """Clé lue hors du code : variable d'environnement, sinon secret Kaggle « GEMINI_API_KEY »
    (menu Add-ons > Secrets du notebook)."""
    cle = P.get("GEMINI_KEY") or os.getenv("GEMINI_API_KEY", "")
    if not cle:                               # v2.9 : comme les autres scripts, param.env du jeu « mziparameter »
        import glob as _g
        pe = next(iter(_cherche("mziparameter/**/param.env")), None)
        if pe:
            for l in open(pe, encoding="utf-8"):
                if l.strip().startswith("GEMINI_API_KEY="):
                    cle = l.split("=", 1)[1].strip().strip('"').strip("'")
    if not cle:
        try:
            from kaggle_secrets import UserSecretsClient
            cle = UserSecretsClient().get_secret("GEMINI_API_KEY") or ""
        except Exception:
            cle = ""
    return cle.strip()


GEMINI_KEY = _cle_gemini() if MODE != "diagnostic" else ""

# ── MESURE DU COÛT GEMINI (jetons renvoyés par l'API après chaque question) ──────────────────────
import threading as _th
COUT = {"questions": 0, "entree": 0, "sortie": 0, "reflexion": 0, "sans_mesure": 0}
COUT_PAR_USAGE = collections.defaultdict(lambda: {"questions": 0, "entree": 0, "sortie": 0, "reflexion": 0})
_cout_verrou = _th.Lock()


def compter(reponse, usage):
    """Ajoute les jetons d'une réponse Gemini (entrée, sortie, réflexion) aux totaux du run."""
    um = getattr(reponse, "usage_metadata", None)
    e = int(getattr(um, "prompt_token_count", 0) or 0)
    so = int(getattr(um, "candidates_token_count", 0) or 0)
    re_ = int(getattr(um, "thoughts_token_count", 0) or 0)
    with _cout_verrou:
        COUT["questions"] += 1
        if um is None:
            COUT["sans_mesure"] += 1
        for cle, v in (("entree", e), ("sortie", so), ("reflexion", re_)):
            COUT[cle] += v; COUT_PAR_USAGE[usage][cle] += v
        COUT_PAR_USAGE[usage]["questions"] += 1


def dollars(c):
    return c["entree"] / 1e6 * PRIX_ENTREE_M + (c["sortie"] + c["reflexion"]) / 1e6 * PRIX_SORTIE_M

# ── 1. CACHE DU SUIVI ─────────────────────────────────────────────────────────────────────────
if MODE != "diagnostic" and not os.path.exists(VIDEO):
    raise SystemExit(f"❌ Vidéo introuvable : {VIDEO}")
# Le cache peut être une archive .zip, ou un dossier (Kaggle DÉCOMPRESSE les .zip ajoutés à un dataset) :
# dans ce cas, le bon dossier est reconnu par son meta.json (chemin de la vidéo du match).
DOSSIER = None


def _dossier_cache(chemin):
    """Dossier contenant meta.json, events.pkl, jersey_map.pkl et frames_data.pkl, à partir d'un dossier (même
    imbriqué) ou d'une archive (même sans l'extension .zip)."""
    if os.path.isfile(chemin) and zipfile.is_zipfile(chemin):
        cible = f"{SORTIE}/cache_{MATCH}"
        with zipfile.ZipFile(chemin) as z:
            z.extractall(cible)
        chemin = cible
    if os.path.isdir(chemin):
        for m in sorted(glob.glob(os.path.join(chemin, "**", "meta.json"), recursive=True)):
            d = os.path.dirname(m)
            if all(os.path.exists(os.path.join(d, f)) for f in ("events.pkl", "jersey_map.pkl", "frames_data.pkl")):
                return d
    return None


if CACHE:
    DOSSIER = _dossier_cache(CACHE)
    if not DOSSIER:
        contenu = sorted(os.listdir(CACHE))[:20] if os.path.isdir(CACHE) else ("fichier" if os.path.exists(CACHE) else "INEXISTANT")
        raise SystemExit(f"❌ CACHE = {CACHE} : aucun cache du suivi lisible (contenu : {contenu})")
    print(f"✅ Cache : {DOSSIER}")
zips = [] if DOSSIER else sorted(_cherche(f"cache_{MATCH}_reference*.zip"))
if DOSSIER:
    pass
elif zips:
    DOSSIER = f"{SORTIE}/cache_{MATCH}"
    with zipfile.ZipFile(zips[-1]) as z:
        z.extractall(DOSSIER)
    print(f"✅ Cache (archive) : {zips[-1]}")
else:
    candidats_c = []
    for m in _cherche("meta.json"):
        d = os.path.dirname(m)
        if not all(os.path.exists(os.path.join(d, f)) for f in ("events.pkl", "jersey_map.pkl", "frames_data.pkl")):
            continue
        try:
            video_meta = str(json.load(open(m)).get("video_path", "")).lower()
        except Exception:
            video_meta = ""
        candidats_c.append((d, video_meta))
        if MATCH.lower() in video_meta or MATCH.lower() in d.lower():
            DOSSIER = d
    if DOSSIER:
        print(f"✅ Cache (dossier décompressé) : {DOSSIER}")
    else:
        print("   caches trouvés en entrée :", [(c[0], os.path.basename(c[1])) for c in candidats_c] or "aucun")
        raise SystemExit(f"❌ Cache du suivi de « {MATCH} » introuvable — ajoute en entrée le dataset qui contient "
                         f"cache_{MATCH}_reference (archive ou dossier avec meta.json, events.pkl, jersey_map.pkl, frames_data.pkl)")


class _S:
    def __init__(self, *a, **k): pass

    def __setstate__(self, st):
        if isinstance(st, dict): self.__dict__.update(st)


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        try: return super().find_class(m, n)
        except Exception: return _S


charger = lambda nom: _U(open(os.path.join(DOSSIER, nom), "rb")).load()
meta = json.load(open(os.path.join(DOSSIER, "meta.json")))
KO, FPS = float(meta["kickoff_offset"]), float(meta["fps"])
jersey_map = {str(k): v for k, v in (charger("jersey_map.pkl") or {}).items()}
events = charger("events.pkl")
frames = charger("frames_data.pkl")
print(f"   coup d'envoi {fmt(KO)} | {len(events)} événements | {len(frames)} images suivies | {len(jersey_map)} numéros lus")
# empreinte du cache : les lectures sauvegardées ne sont réutilisées que pour CE cache
SIGNATURE = f"{os.path.basename(str(meta.get('video_path', '')))}|{KO:.3f}|{FPS:.3f}|{len(frames)}"



# ── POSITIONS ET SEGMENTS (CPU, aucun appel) ──────────────────────────────────────────────────
def _tsu(p):
    v = p.get("time_since_update")
    try:
        return None if v is None else int(v)
    except Exception:
        return None


N_POS = N_TSU = N_PREDITES = N_HORS = N_SANS_TID = 0
for f in frames:
    for p in f.get("players", []):
        N_POS += 1
        if _tsu(p) is not None:
            N_TSU += 1
AVEC_TSU = N_TSU > 0
if AVEC_TSU:
    print(f"✅ time_since_update présent ({100 * N_TSU / max(1, N_POS):.1f} % des positions) : "
          f"seules les vraies détections (== 0) sont utilisées")
else:
    msg = ("time_since_update ABSENT de ce cache (cache antérieur au replay.py du 28/10) : les positions PRÉDITES "
           "par le suivi ne peuvent pas être écartées. En mode « historique », le résultat serait peu sûr.")
    if EXIGER_TSU:
        raise SystemExit("❌ " + msg)
    print("⚠️ " + msg)

pistes = collections.defaultdict(dict)           # tracker_id -> {image suivie: (cadre, équipe du cache)}
for f in frames:
    num = int(f.get("frame", 0))
    for p in f.get("players", []):
        tid, b = p.get("tracker_id"), p.get("bbox")
        if tid is None:
            N_SANS_TID += 1; continue
        if AVEC_TSU and _tsu(p) != 0:
            N_PREDITES += 1; continue
        if b is None or not dans_image(b):
            N_HORS += 1; continue
        if num in pistes[str(tid)]:
            continue
        pistes[str(tid)][num] = ([float(v) for v in b], p.get("team"))
if not pistes:
    raise SystemExit("❌ Ce cache ne contient pas l'identifiant DeepSort (tracker_id) : refais le suivi avec la version actuelle")


def decouper(pos):
    """Coupe une piste aux ruptures probables d'identité. Renvoie [(liste de numéros d'image, raison de la
    coupure qui PRÉCÈDE ce segment ou None)]."""
    nums = sorted(pos)
    segs, courant, raison = [], [nums[0]], None
    for a, b in zip(nums, nums[1:]):
        dt = (b - a) / FPS
        ba, bb = pos[a][0], pos[b][0]
        ha, hb = max(1.0, ba[3] - ba[1]), max(1.0, bb[3] - bb[1])
        d = np.hypot((ba[0] + ba[2]) / 2 - (bb[0] + bb[2]) / 2, ba[3] - bb[3]) / ((ha + hb) / 2)
        r = None
        if dt > COUPURE_TROU_S:
            r = f"trou {dt:.1f} s"
        elif d > SAUT_BASE_H + SAUT_H_PAR_S * dt:
            r = f"saut {d:.1f} h en {dt:.2f} s"
        elif max(hb / ha, ha / hb) > RAPPORT_TAILLE_MAX:
            r = f"taille ×{max(hb / ha, ha / hb):.1f}"
        if r:
            segs.append((courant, raison)); courant, raison = [b], r
        else:
            courant.append(b)
    segs.append((courant, raison))
    return segs


segments = {}                                    # « tid@début-fin » -> {"tid", "pos", "raison"}
raisons = collections.Counter()
for tid, pos in pistes.items():
    for nums, raison in decouper(pos):
        sid = f"{tid}@{nums[0]}-{nums[-1]}"
        segments[sid] = {"tid": tid, "pos": {n: pos[n] for n in nums}, "raison": raison}
        if raison:
            raisons[raison.split()[0]] += 1
print(f"   {len(pistes)} pistes DeepSort → {len(segments)} segments | coupures : {dict(raisons)}"
      + (f" | positions prédites écartées : {N_PREDITES}" if AVEC_TSU else "")
      + f" | hors image : {N_HORS}" + (f" | sans tracker_id : {N_SANS_TID}" if N_SANS_TID else ""))

debut = lambda sid: KO + min(segments[sid]["pos"]) / FPS
fin = lambda sid: KO + max(segments[sid]["pos"]) / FPS


# ── DIAGNOSTIC (MODE = "diagnostic") ──────────────────────────────────────────────────────────
def _court(v, n=90):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "…"


def diagnostic():
    """Mesures seulement, aucune conclusion automatique."""
    print(f"\n{'─' * 90}\nDIAGNOSTIC DU CACHE (CPU, aucun appel Gemini, vidéo non lue)\n{'─' * 90}")
    champs = collections.Counter(k for f in frames for p in f.get("players", []) for k in p)
    print("1) champs d'une position :", ", ".join(f"{k} {n}" for k, n in champs.most_common()))
    nums = sorted({int(f.get("frame", 0)) for f in frames})
    ecarts = collections.Counter(np.diff(nums).tolist()) if len(nums) > 1 else collections.Counter()
    if ecarts:
        typique = ecarts.most_common(1)[0][0] / FPS
        print(f"2) écart typique entre images suivies : {typique:.2f} s → TENUE_REPERE_S = {TENUE_REPERE_S} s "
              + ("suffisante" if TENUE_REPERE_S >= 1.5 * typique else f"trop courte, prévoir ≥ {1.5 * typique:.2f} s"))
    tsu = collections.Counter(_tsu(p) for f in frames for p in f.get("players", []))
    print(f"3) {N_POS} positions ({N_POS / max(1, len(frames)):.1f} par image suivie) | time_since_update : "
          + (", ".join(f"{k}: {v}" for k, v in sorted(tsu.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))[:8])
             if AVEC_TSU else "ABSENT"))
    fiable = collections.Counter(p.get("fiable") for f in frames[::50] for p in f.get("players", []))
    print(f"   « fiable » (échantillon) : {dict(fiable)} (non utilisé : même critère que time_since_update)")
    durees = np.array([(max(s["pos"]) - min(s["pos"])) / FPS for s in segments.values()])
    print(f"4) segments : {len(segments)} | durée médiane {np.median(durees):.1f} s | ≥ {MIN_DUREE_S} s : "
          f"{int((durees >= MIN_DUREE_S).sum())} | coupures : {dict(raisons)}")
    print(f"\n5) pistes examinées : {TIDS_DIAG}")
    for tid in map(str, TIDS_DIAG):
        sids = sorted((s for s, v in segments.items() if v["tid"] == tid), key=debut)
        if not sids:
            print(f"   piste {tid} : absente"); continue
        print(f"   piste {tid} : {len(sids)} segment(s)")
        for s in sids:
            v = segments[s]
            eq = collections.Counter(e for _, e in v["pos"].values())
            hmax = max(b[3] - b[1] for b, _ in v["pos"].values())
            print(f"     {fmt(debut(s))} → {fmt(fin(s))} ({fin(s) - debut(s):4.1f} s, {len(v['pos']):3d} pos., "
                  f"hauteur max {hmax:.0f} px) | équipe du cache {dict(eq)}"
                  + (f" | coupé avant : {v['raison']}" if v["raison"] else ""))
    print(f"\n✅ Diagnostic terminé en {time.time() - T0:.0f} s.")


if MODE == "diagnostic":
    diagnostic()
    raise SystemExit(0)

# ── LECTURES SAUVEGARDÉES : FUSION DE TOUS LES FICHIERS (format 2) ─────────────────────────────
FICHIER_LECTURES = FICHIER_LECTURES or f"{SORTIE}/lectures_v30_{MATCH}_{PARSEQ_VERSION or 'auto'}.json"
VERSION = "v2.8"


def fichiers_lectures():
    """Du plus récent au plus ancien : v2.7 à v2.2 ; pour chacun /kaggle/working puis /kaggle/input."""
    out = [FICHIER_LECTURES]
    for v in ("v30",):                              # v3.0 : uniquement les lectures PARSeq
        out += [f"{SORTIE}/lectures_{v}_{MATCH}.json"] + sorted(_cherche(f"lectures_{v}_{MATCH}.json"))
    return [c for c in dict.fromkeys(out) if os.path.exists(c)]


def _lue(e):
    return e.get("numeros") is not None and not e.get("non_lu")


def fusionner_lectures():
    lectures, coul_r, conflits, sources, sauve = {}, {}, [], [], None
    for c in fichiers_lectures():
        try:
            d = json.load(open(c, encoding="utf-8"))
        except Exception as e:
            print(f"   ⚠️ {c} illisible ({e}) : ignoré"); continue
        if d.get("format") != 2 or d.get("match") != MATCH or d.get("signature") != SIGNATURE or d.get("modele") != MODELE:
            print(f"   ⚠️ {c} : autre format, match, cache ou modèle : ignoré"); continue
        nom = os.path.basename(c)
        if sauve is None or (sauve.get("couleurs_source") != "gemini" and d.get("couleurs_source") == "gemini"):
            sauve = d                                    # couleurs des équipes : fichier le plus récent issu de Gemini
        n_ajout = 0
        for sid, e in (d.get("segments") or {}).items():
            e = dict(e); e.setdefault("source", nom)
            deja = lectures.get(sid)
            if deja is None:
                lectures[sid] = e; n_ajout += 1
            elif _lue(e) and not _lue(deja):
                lectures[sid] = e; n_ajout += 1          # une vraie lecture l'emporte sur « évitée » ou « plafond »
            elif _lue(e) and _lue(deja) and (e.get("numeros") != deja.get("numeros") or e.get("choix") != deja.get("choix")):
                conflits.append({"segment": sid, "gardee": deja.get("source"), "numeros_gardes": deja.get("numeros"),
                                 "ecartee": e.get("source"), "numeros_ecartes": e.get("numeros"),
                                 "meme_images": e.get("choix") == deja.get("choix")})
        for sid, e in (d.get("raccord_couleurs") or {}).items():
            coul_r.setdefault(sid, e)
        sources.append((nom, c, n_ajout))
    return lectures, coul_r, conflits, sources, sauve


lectures, COUL_R, CONFLITS, SOURCES_L, sauve = fusionner_lectures()
for nom, c, n in SOURCES_L:
    print(f"✅ Lectures : {c} → {n} segment(s) apportés")
if SOURCES_L:
    print(f"   fusion : {len(lectures)} segment(s) connus, dont {sum(1 for e in lectures.values() if _lue(e))} lus par Gemini | "
          f"conflits entre fichiers : {len(CONFLITS)} (le plus récent gardé, l'autre journalisé)")
with open(f"{SORTIE}/lectures_conflits_v28_{MATCH}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["segment", "gardee", "numeros_gardes", "ecartee", "numeros_ecartes", "meme_images"], delimiter=";")
    w.writeheader(); w.writerows(CONFLITS)


def sauver_lectures():
    d = {"format": 2, "match": MATCH, "signature": SIGNATURE, "modele": MODELE,
         "couleurs": {str(k): v for k, v in COULEUR_EQUIPE.items()}, "couleurs_source": SOURCE_COULEURS,
         "segments": lectures, "raccord_couleurs": COUL_R, "conflits": CONFLITS}
    tmp = FICHIER_LECTURES + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, FICHIER_LECTURES)


# ── 2. COULEUR DE CHAQUE ÉQUIPE ───────────────────────────────────────────────────────────────
def nom_couleur(h, s, v):
    """Teinte OpenCV (0-179), saturation et valeur (0-255) -> nom de couleur en français."""
    if v < 60:                    return "noir"
    if s < 45:                    return "blanc" if v > 165 else "gris"
    if h < 9 or h >= 170:         return "rouge"
    if h < 22:                    return "orange"
    if h < 34:                    return "jaune"
    if h < 85:                    return "vert"
    if h < 100:                   return "bleu clair"
    if h < 130:                   return "bleu"
    return "violet"


COULEURS_VALIDES = ["rouge", "bleu", "bleu clair", "blanc", "noir", "jaune", "vert", "orange", "violet", "gris"]


AUTRES_COULEURS = {"gardiens": collections.Counter(), "arbitre": collections.Counter()}   # 19/11


def couleurs_par_gemini():
    """Même source que la pré-analyse du site (Gemini, juste après le coup d'envoi), mais RATTACHÉE aux équipes du
    suivi : les joueurs de l'équipe 0 sont encadrés en cyan « A », ceux de l'équipe 1 en magenta « B »."""
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        print(f"   (Gemini indisponible : {e})"); return {}
    client = genai.Client(api_key=GEMINI_KEY)
    par_frame = {f.get("frame", 0): f for f in frames}
    cles = np.array(sorted(par_frame))
    votes = {0: collections.Counter(), 1: collections.Counter()}
    cap_g = cv2.VideoCapture(VIDEO)
    for dt in (10, 30, 60):
        num = int(cles[np.argmin(np.abs(cles - dt * FPS))])
        f = par_frame[num]
        cap_g.set(cv2.CAP_PROP_POS_MSEC, (KO + num / FPS) * 1000)
        ok, img = cap_g.read()
        if not ok:
            continue
        n = {0: 0, 1: 0}
        for p in f.get("players", []):
            b = p.get("bbox"); eq_ = p.get("team")
            # vraies détections seulement quand le cache le permet (time_since_update == 0) ; « conf » vaut toujours 1
            if eq_ not in (0, 1) or b is None or (AVEC_TSU and _tsu(p) != 0) or not dans_image(b):
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            c = (255, 255, 0) if eq_ == 0 else (255, 0, 255)                    # BGR : cyan / magenta
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
            cv2.putText(img, "A" if eq_ == 0 else "B", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 3)
            n[eq_] += 1
        if min(n.values()) < 2:
            print(f"   Gemini, coup d'envoi + {dt} s : image ignorée (moins de 2 joueurs encadrés par équipe : {n})")
            continue
        cv2.imwrite(f"{SORTIE}/equipes_ko_plus_{dt}s.jpg", img)
        _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        consigne = ("Football match image. Players of one team are marked with a CYAN box labelled A; players of the "
                    "other team with a MAGENTA box labelled B. What is the main SHIRT color of team A and of team B? "
                    "Also give the shirt color of each GOALKEEPER you can see (they wear a different color from their "
                    "team), with the team they belong to if you can tell, and the shirt color of the REFEREE. "
                    "Ignore the box colors themselves. Answer ONLY with JSON: {\"A\": \"...\", \"B\": \"...\", "
                    "\"gardiens\": [{\"couleur\": \"...\", \"equipe\": \"A|B|?\"}], \"arbitre\": \"...\"} using these "
                    "French names: " + ", ".join(COULEURS_VALIDES) + " (\"?\" if not visible).")
        try:
            rep_g = client.models.generate_content(model=MODELE,
                                                   contents=[consigne, types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")],
                                                   config={"temperature": 0.0})
            compter(rep_g, "couleurs des équipes")
            txt = rep_g.text or ""
            d = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
            for eq_, k in ((0, "A"), (1, "B")):
                c = str(d.get(k, "")).strip().lower()
                if c in COULEURS_VALIDES:
                    votes[eq_][c] += 1
            print(f"   Gemini, coup d'envoi + {dt} s : A (équipe 0) « {d.get('A')} », B (équipe 1) « {d.get('B')} »"
                  f" | gardiens {d.get('gardiens')} | arbitre « {d.get('arbitre')} »")
            for _g in (d.get("gardiens") or []):
                if isinstance(_g, dict) and str(_g.get("couleur", "")).lower() in COULEURS_VALIDES:
                    AUTRES_COULEURS["gardiens"][(str(_g["couleur"]).lower(), str(_g.get("equipe", "?")))] += 1
            if str(d.get("arbitre", "")).lower() in COULEURS_VALIDES:
                AUTRES_COULEURS["arbitre"][str(d["arbitre"]).lower()] += 1
        except Exception as e:
            print(f"   Gemini, coup d'envoi + {dt} s : réponse inexploitable ({e})")
    cap_g.release()
    return {eq_: v.most_common(1)[0][0] for eq_, v in votes.items() if v}


COULEUR_EQUIPE, SOURCE_COULEURS = {}, ""
if sauve and sauve.get("couleurs_source") == "gemini" and len(sauve.get("couleurs") or {}) == 2 and not GARDIEN:
    COULEUR_EQUIPE = {int(k): v for k, v in sauve["couleurs"].items()}
    SOURCE_COULEURS = "gemini"
    print("   couleurs des équipes réutilisées (Gemini, sauvegarde) :", COULEUR_EQUIPE)
elif not GEMINI_KEY:
    raise SystemExit("❌ Clé Gemini absente (secret Kaggle « GEMINI_API_KEY » introuvable ou non coché pour ce notebook, "
                     "menu Add-ons > Secrets). Les couleurs des équipes viennent de Gemini ; la mesure par pixels, en "
                     "secours, n'est pas fiable (Franchimont, 29/10 : les deux équipes mesurées « rouge »).")
else:
    COULEUR_EQUIPE = couleurs_par_gemini()
    if len(COULEUR_EQUIPE) == 2:
        SOURCE_COULEURS = "gemini"
        for eq_, c in sorted(COULEUR_EQUIPE.items()):
            print(f"   équipe {eq_} : maillots « {c} » (Gemini, juste après le coup d'envoi)")

echantillons = collections.defaultdict(list)
cap = cv2.VideoCapture(VIDEO) if len(COULEUR_EQUIPE) < 2 else None
pas = max(1, len(frames) // 60)
for f in (frames[::pas] if cap is not None else []):
    # cadres ENTIÈREMENT dans l'image, et vraies détections quand time_since_update existe
    joueurs = [p for p in f.get("players", []) if p.get("team") in (0, 1) and p.get("bbox") is not None
               and not (AVEC_TSU and _tsu(p) != 0) and dans_image(p["bbox"])]
    if not joueurs:
        continue
    cap.set(cv2.CAP_PROP_POS_MSEC, (KO + f.get("frame", 0) / FPS) * 1000)
    ok, img = cap.read()
    if not ok:
        continue
    H, W = img.shape[:2]
    for p in joueurs:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        h = y2 - y1
        if h < 60:
            continue
        haut = img[max(0, y1 + int(0.15 * h)):min(H, y1 + int(0.45 * h)), max(0, x1 + (x2 - x1) // 5):min(W, x2 - (x2 - x1) // 5)]
        if haut.size == 0:
            continue
        hsv = cv2.cvtColor(haut, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
        # pixels de PELOUSE exclus (teintes vertes saturées) ; puis CHAQUE pixel reçoit un nom de couleur et le
        # joueur prend la couleur majoritaire
        pelouse = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 40)
        maillot = hsv[~pelouse]
        if len(maillot) < 0.2 * len(hsv) or len(maillot) < 30:
            continue
        noms = collections.Counter(nom_couleur(h_, s_, v_) for h_, s_, v_ in maillot)
        echantillons[p["team"]].append(noms.most_common(1)[0][0])
if cap is not None:
    cap.release()
    print("   (mesure par pixels, en secours)")
    COULEUR_EQUIPE, SOURCE_COULEURS = {}, "pixels"
    if not echantillons:
        print("   ⚠️ Aucun échantillon de maillot exploitable (équipe en VERT, ou joueurs trop petits) : couleurs inconnues")
for eq_, lst in sorted(echantillons.items()):
    if lst:
        votes = collections.Counter(lst)
        COULEUR_EQUIPE[eq_] = votes.most_common(1)[0][0]
        print(f"   équipe {eq_} : maillots « {COULEUR_EQUIPE[eq_]} » ({len(lst)} joueurs mesurés : "
              + ", ".join(f"{c} {100 * n / len(lst):.0f} %" for c, n in votes.most_common(3)) + ")")

# ── 3. SEGMENTS CANDIDATS ─────────────────────────────────────────────────────────────────────
FAMILLES = [{"rouge", "orange"}, {"bleu", "bleu clair", "violet"}, {"blanc", "gris"}, {"jaune", "orange"}]
famille = lambda c: set().union({c}, *[fam for fam in FAMILLES if c in fam])
if EQUIPE in (0, 1):
    equipes_voulues = [EQUIPE]
else:
    proches = lambda a, b: a == b or any(a in fam and b in fam for fam in FAMILLES)
    equipes_voulues = [e for e, c in COULEUR_EQUIPE.items() if c == COULEUR.lower()] or \
                      [e for e, c in COULEUR_EQUIPE.items() if proches(c, COULEUR.lower())]
if len(equipes_voulues) != 1 or equipes_voulues[0] not in COULEUR_EQUIPE:
    raise SystemExit(f"❌ Équipe « {COULEUR} » introuvable ou ambiguë (couleurs : {COULEUR_EQUIPE}) — règle EQUIPE = 0 ou 1")
EQ = equipes_voulues[0]
if len(COULEUR_EQUIPE) < 2:
    print(f"   ⚠️ couleur de l'équipe adverse inconnue : une vignette est conforme si au moins {100 * SEUIL_COULEUR:.0f} % "
          f"du haut du maillot est « {COULEUR_EQUIPE[EQ]} » (moins discriminant)")
elif famille(COULEUR_EQUIPE[0]) & famille(COULEUR_EQUIPE[1]):
    print(f"   ⚠️ couleurs d'équipe proches ({COULEUR_EQUIPE}) : la mesure de couleur sera peu discriminante")

_FEN = [(int(h.split(":")[0]) * 60 + int(h.split(":")[1]) - FENETRE_AVANT,
         int(h.split(":")[0]) * 60 + int(h.split(":")[1]) + FENETRE_APRES) for h in FENETRES]
if BOUT_EN_BOUT and BOUT_EN_BOUT.get("bornes"):     # joueur seul : seulement le temps de jeu (KO1→fin 1re MT, KO2→fin)
    _FEN = [(float(a), float(b)) for a, b in BOUT_EN_BOUT["bornes"]]
    print(f"   temps de jeu lu : " + " ; ".join(f"{fmt(a)}-{fmt(b)}" for a, b in _FEN))
candidats = []
for sid, v in segments.items():
    if _FEN and not any(debut(sid) <= b and fin(sid) >= a for a, b in _FEN):
        continue
    if MODE == "essai" and v["tid"] not in set(map(str, TIDS_DIAG)):
        continue
    pos = v["pos"]
    nums = sorted(pos)
    lisibles = [n for n in nums if pos[n][0][3] - pos[n][0][1] >= MIN_HAUTEUR]
    if (nums[-1] - nums[0]) / FPS < MIN_DUREE_S or len(lisibles) < 2:
        continue
    part_eq = sum(1 for _, e in pos.values() if e == EQ) / len(pos)
    if MODE == "complet" and part_eq < PREFILTRE_EQUIPE:
        continue
    choix = [lisibles[int(round(q * (len(lisibles) - 1)))] for q in np.linspace(0, 1, min(N_LECTURES, len(lisibles)))]
    candidats.append((max(pos[n][0][3] - pos[n][0][1] for n in nums), sid, list(dict.fromkeys(choix))))
candidats.sort(reverse=True)
N_CANDIDATS_TOTAL = len(candidats)
if MODE == "complet" and MAX_SEGMENTS:
    candidats = candidats[:MAX_SEGMENTS]
print(f"\n   équipe {EQ} ({COULEUR_EQUIPE[EQ]}) | {len(candidats)} segment(s) à examiner sur {N_CANDIDATS_TOTAL}"
      + (f" (pistes {TIDS_DIAG} seulement, mode essai)" if MODE == "essai" else
         f" (≥ {MIN_DUREE_S} s, ≥ 2 images lisibles, ≥ {100 * PREFILTRE_EQUIPE:.0f} % de positions de l'équipe dans le cache)"))



# ── 4. COULEUR MESURÉE (méthode 2, inchangée depuis la v2.2) ──────────────────────────────────
def noms_pixels(hsv):
    """Même découpage que nom_couleur, vectorisé."""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    return np.select([v < 60, (s < 45) & (v > 165), s < 45, (h < 9) | (h >= 170), h < 22, h < 34, h < 85, h < 100, h < 130],
                     ["noir", "blanc", "gris", "rouge", "orange", "jaune", "vert", "bleu clair", "bleu"], default="violet")


METHODE_COULEUR = 2      # 2 (29/10) : couleur STRICTE de l'équipe (le rouge ne compte plus l'orange, qui captait
                         # le jaune de l'arbitre) et pixels de pelouse exclus d'après la pelouse autour du joueur
MESURE_FAMILLES = [{"bleu", "bleu clair"}, {"blanc", "gris"}]
_nom_mesure_base = lambda c: set().union({c}, *[fam for fam in MESURE_FAMILLES if c in fam])
# v3.2 (07/11) : un maillot BLEU CIEL pâle est vu en grande partie « blanc / gris » par la mesure (peu saturé) ;
# 484 segments sur 661 du n°16 étaient écartés par la couleur. On compte aussi blanc et gris pour une équipe
# « bleu clair », SEULEMENT si l'autre équipe n'est ni blanche ni grise (sinon on ne pourrait plus les distinguer).
COULEUR_PALE_ELARGIE = True


def nom_mesure(c):
    fam = _nom_mesure_base(c)
    autres = [v for v in COULEUR_EQUIPE.values() if v != c]
    if COULEUR_PALE_ELARGIE and c == "bleu clair" and not any(v in ("blanc", "gris") for v in autres):
        fam = fam | {"blanc", "gris"}
    return fam


def _hsv(zone):
    return cv2.cvtColor(zone, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int) if zone.size else np.zeros((0, 3), int)


def parts_couleur(img, b):
    """Part des pixels du haut du maillot dans la couleur de chaque équipe : [part 0, part 1].
    Pixels de pelouse retirés : référence = pixels verts juste sous les pieds et sur les côtés du cadre."""
    Hh, Ww = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in b]
    h, w = y2 - y1, x2 - x1
    haut = img[max(0, y1 + int(0.15 * h)):max(0, y1 + int(0.45 * h)), max(0, x1 + w // 5):max(0, x2 - w // 5)]
    if haut.size == 0:
        return [0.0, 0.0]
    hsv = _hsv(haut)
    noms = noms_pixels(hsv)
    autour = np.vstack([_hsv(img[min(Hh, y2):min(Hh, y2 + max(4, h // 10)), max(0, x1):max(0, x2)]),
                        _hsv(img[max(0, y1):max(0, y2), max(0, x1 - w // 6):max(0, x1)]),
                        _hsv(img[max(0, y1):max(0, y2), min(Ww, x2):min(Ww, x2 + w // 6)])])
    garde = np.ones(len(noms), bool)
    if len(autour):
        gazon_ref = autour[noms_pixels(autour) == "vert"]
        if len(gazon_ref) > 30:
            m = np.median(gazon_ref, axis=0)
            garde = ~((np.abs(hsv[:, 0] - m[0]) <= 8) & (np.abs(hsv[:, 1] - m[1]) <= 60) & (np.abs(hsv[:, 2] - m[2]) <= 45))
    if garde.mean() < 0.25:
        return [0.0, 0.0]                         # presque que de la pelouse : cadre vide ou joueur minuscule
    noms = noms[garde]
    return [round(float(np.isin(noms, list(nom_mesure(COULEUR_EQUIPE[e]))).mean()), 3) if e in COULEUR_EQUIPE else 0.0
            for e in (0, 1)]


def equipe_mesuree(parts):
    """0, 1, ou None (arbitre, adversaire collé, ombre : indécis)."""
    a, b = parts
    if max(a, b) < SEUIL_COULEUR or max(a, b) < 1.5 * min(a, b):
        return None
    return 0 if a > b else 1


# ── 4. VIGNETTES, COULEUR MESURÉE AVANT LA QUESTION, LECTURE DU NUMÉRO ────────────────────────
import threading, bisect
_tl = threading.local()
_caps, _caps_verrou = [], threading.Lock()


def image_video(num):
    """Image suivie n° num ; une lecture de la vidéo par fil (lectures en parallèle)."""
    c = getattr(_tl, "cap", None)
    if c is None:
        c = cv2.VideoCapture(VIDEO); _tl.cap = c
        with _caps_verrou:
            _caps.append(c)
    c.set(cv2.CAP_PROP_POS_MSEC, (KO + num / FPS) * 1000)
    return c.read()


def decoupe(img, b):
    x1, y1, x2, y2 = [int(v) for v in b]; mx, my = (x2 - x1) // 6, (y2 - y1) // 10
    c = img[max(0, y1 - my):min(1080, y2 + my), max(0, x1 - mx):min(1920, x2 + mx)]
    return cv2.resize(c, (int(256 * c.shape[1] / c.shape[0]), 256)) if c.size else None


def vignettes(sid, choix):
    ims, parts, images = [], [], []
    for num in choix:
        b = segments[sid]["pos"][num][0]
        ok, img = image_video(num)
        if not ok:
            continue
        c = decoupe(img, b)
        if c is not None:
            ims.append(c); parts.append(parts_couleur(img, b)); images.append(int(num))
    return ims, parts, images


SEUIL_PARSEQ = 0.80
import sys as _sys, subprocess as _sp
try:
    import timm, pytorch_lightning, nltk  # noqa: F401  (production : à installer avec requirements.txt)
except ImportError:
    _sp.run([_sys.executable, "-m", "pip", "install", "-q", "timm", "pytorch-lightning", "nltk"], check=False)
import torch as _torch
from torchvision import transforms as _T
# 06/11 : le modèle le plus récent est pris d'abord (v2 multi-matchs), sinon v1 ; code PARSeq FIGÉ (commit)
PARSEQ_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
_poids = P.get("PARSEQ_POIDS") if P.get("PARSEQ_POIDS") and os.path.exists(P["PARSEQ_POIDS"]) else None
for _v in (() if _poids else ((PARSEQ_VERSION,) if PARSEQ_VERSION else ("v2", "v1"))):
    _poids = next(iter(_cherche(f"parseq_scoutia_maillots_{_v}.pt")
                       + glob.glob(f"{SORTIE}/parseq_scoutia_maillots_{_v}.pt")), None)
    if _poids:
        break
if not _poids:
    raise SystemExit("❌ parseq_scoutia_maillots_v2.pt ou _v1.pt introuvable dans les entrées")
_dev = "cuda" if _torch.cuda.is_available() else "cpu"
PARSEQ = _torch.hub.load(f"baudm/parseq:{PARSEQ_COMMIT}", "parseq", pretrained=True, trust_repo=True)
PARSEQ.load_state_dict(_torch.load(_poids, map_location="cpu")); PARSEQ = PARSEQ.eval().to(_dev)
_tfp = _T.Compose([_T.ToPILImage(), _T.Resize((32, 128), _T.InterpolationMode.BICUBIC), _T.ToTensor(), _T.Normalize(0.5, 0.5)])
_verrou_parseq = threading.Lock()
_itos = getattr(PARSEQ.tokenizer, "_itos", None)          # 06/11 : lecture restreinte aux chiffres 0-9
_MASQUE = (_torch.tensor([0.0 if (i == 0 or str(c).isdigit()) else float("-inf") for i, c in enumerate(_itos)], device=_dev)
           if _itos else None)
print(f"   lecteur de numéros : PARSeq affiné ({_poids}) sur {_dev}, seuil {SEUIL_PARSEQ}")


def _decoupes_dos(im, b):
    x1, y1, x2, y2 = [int(v) for v in b]; h, w = y2 - y1, x2 - x1
    if h < 20 or w < 8:
        return []
    H, W = im.shape[:2]; out = []
    for a, z in ((0.12, 0.55), (0.18, 0.50)):        # identiques à l'apprentissage
        c = im[max(0, y1 + int(a * h)):min(H, y1 + int(z * h)), max(0, x1 - int(0.1 * w)):min(W, x2 + int(0.1 * w))]
        if c.size:
            k = 64 / max(1, c.shape[0])
            out.append(cv2.resize(c, (max(16, int(c.shape[1] * k)), 64), interpolation=cv2.INTER_CUBIC))
    return out


def lire_parseq(dos_par_image):
    """dos_par_image : [[découpes de l'image 1], …] → [numéro ou None par image]"""
    plat = [(i, c) for i, cs in enumerate(dos_par_image) for c in cs]
    res = [None] * len(dos_par_image)
    if not plat:
        return res
    with _verrou_parseq, _torch.no_grad():
        x = _torch.stack([_tfp(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for _, c in plat]).to(_dev)
        _lg = PARSEQ(x)
        if _MASQUE is not None and _lg.shape[-1] == _MASQUE.shape[0]:
            _lg = _lg + _MASQUE
        lab, conf = PARSEQ.tokenizer.decode(_lg.softmax(-1))
    meilleur = {}
    for (i, _), l, cf in zip(plat, lab, conf):
        cf = float(cf.prod()); d = "".join(ch for ch in l if ch.isdigit())
        if cf >= SEUIL_PARSEQ and 1 <= len(d) <= 2 and (i not in meilleur or cf > meilleur[i][1]):
            meilleur[i] = (int(d), cf)
    for i, (n, _) in meilleur.items():
        res[i] = n
    return res


compteur = {"questions": 0, "evitees": 0, "plafond": 0}
verrou_compteur = threading.Lock()


def lire(sid, choix):
    ims, parts, images = vignettes(sid, choix)
    entree = {"choix": [int(n) for n in choix], "images": images, "parts": parts, "numeros": None,
              "couleur_v": METHODE_COULEUR, "source": f"lectures_v30_{MATCH}.json",
              "reflexion": REFLEXION_LECTURE or "normale"}
    if sum(1 for p in parts if equipe_mesuree(p) == EQ) < 1:          # 07/11 : lu dès 1 vignette de la bonne couleur (gratuit) ; règle de retenue inchangée
        entree["numeros"], entree["non_lu"] = [None] * len(ims), "couleur"     # ne peut pas être retenu : pas de question
        with verrou_compteur:
            compteur["evitees"] += 1
        return sid, entree, ims
    with verrou_compteur:
        if compteur["questions"] >= MAX_QUESTIONS:
            compteur["plafond"] += 1
            entree["numeros"], entree["non_lu"] = [None] * len(ims), "plafond"   # couleur gardée, lu au run suivant
            return sid, entree, ims
        compteur["questions"] += 1
    dos = []
    for num in images:                                           # v3.0 : lecture par PARSeq affiné, sur le dos
        ok, img = image_video(num)
        dos.append(_decoupes_dos(img, segments[sid]["pos"][num][0]) if ok else [])
    try:
        entree["numeros"] = lire_parseq(dos)
        entree["source"] = f"lectures_v30_{MATCH}.json (PARSeq affiné)"
        return sid, entree, ims
    except Exception as e:
        print(f"   ⚠️ lecture PARSeq {sid} : {e}")
        return sid, None, ims


def a_relire(sid, choix):
    e = lectures.get(sid)
    if not e or e.get("choix") != choix:
        return True
    if e.get("non_lu") == "plafond":
        return True                                                         # question pas encore posée
    return bool(e.get("non_lu")) and e.get("couleur_v") != METHODE_COULEUR   # couleur à remesurer avec la méthode actuelle


a_lire = [c for c in candidats if a_relire(c[1], c[2])]
_a_lire = {c[1] for c in a_lire}
N_REUTILISEES_ICI = sum(1 for _, sid, _ in candidats if sid not in _a_lire and _lue(lectures.get(sid) or {}))
N_EVITEES_CONNUES = sum(1 for _, sid, _ in candidats if sid not in _a_lire and (lectures.get(sid) or {}).get("non_lu") == "couleur")
VIGN, erreurs = {}, set()
print(f"   lectures : {len(candidats) - len(a_lire)} segment(s) déjà connu(s) | {len(a_lire)} à examiner "
      f"(couleur d'abord ; question seulement si ≥ 2 vignettes « {COULEUR_EQUIPE[EQ]} », au plus {MAX_QUESTIONS})")
# ── 09/11 : LECTURE RAPIDE ─────────────────────────────────────────────────────────────────────────────────────
# La vidéo est lue UNE fois, dans l'ordre (plus de saut d'une image à l'autre), en 2 processus (une moitié du temps
# chacun) qui découpent les vignettes, mesurent la couleur et préparent les découpes du dos ; PARSeq lit ensuite tout
# sur le GPU par lots. TOUS les morceaux candidats sont lus (les deux équipes) : le fichier de lectures sert ensuite
# au montage de N'IMPORTE QUEL joueur du match sans relire la vidéo. Retenue d'un morceau : règle inchangée.
LECTURE_RAPIDE = True
N_PROCESSUS_LECTURE = 2


def _lire_tranche(taches):
    """taches : [(n° d'image du suivi, [(sid, idx, cadre), …]), …] triées → [(sid, idx, n°, parts, [dos jpg])]"""
    cv2.setNumThreads(1)                            # processus fils : pas de sur-parallélisme
    cap = cv2.VideoCapture(VIDEO)
    fps_v = cap.get(cv2.CAP_PROP_FPS) or FPS
    out, cur = [], None
    for num, items in taches:
        cible = int(round((KO + num / FPS) * fps_v))
        if cur is None or cible < cur or cible - cur > 300:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cible); cur = cible
        while cur < cible:
            cap.grab(); cur += 1
        ok, img = cap.read(); cur += 1
        if not ok:
            continue
        for sid, idx, b in items:
            dos = [cv2.imencode(".png", d)[1].tobytes() for d in _decoupes_dos(img, b)]
            out.append((sid, idx, int(num), parts_couleur(img, b), dos))
    cap.release()
    return out


if a_lire and LECTURE_RAPIDE:
    import multiprocessing as _mp
    t_l = time.time()
    _besoin = collections.defaultdict(list)
    for _, sid, choix in a_lire:
        for idx, num in enumerate(choix):
            _besoin[int(num)].append((sid, idx, segments[sid]["pos"][num][0]))
    _taches = sorted(_besoin.items())
    _n = max(1, N_PROCESSUS_LECTURE)
    _morceaux = [_taches[i * len(_taches) // _n:(i + 1) * len(_taches) // _n] for i in range(_n)]
    with _mp.get_context("fork").Pool(_n) as _pool:
        _res = [r for part in _pool.map(_lire_tranche, _morceaux) for r in part]
    t_dec = time.time() - t_l
    _par_seg = collections.defaultdict(dict)
    for sid, idx, num, parts, dos in _res:
        _par_seg[sid][idx] = (num, parts, [cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR) for d in dos])
    # PARSeq par lots (toutes les images de tous les morceaux)
    _ordre = [(sid, idx) for sid in _par_seg for idx in sorted(_par_seg[sid])]
    _lus = {}
    for k in range(0, len(_ordre), 400):
        lot = _ordre[k:k + 400]
        nums = lire_parseq([_par_seg[sid][idx][2] for sid, idx in lot])
        for (sid, idx), n in zip(lot, nums):
            _lus[(sid, idx)] = n
    for _, sid, choix in a_lire:
        d = _par_seg.get(sid, {})
        idxs = sorted(d)
        lectures[sid] = {"choix": [int(n) for n in choix], "images": [d[i][0] for i in idxs], "parts": [d[i][1] for i in idxs],
                         "numeros": [_lus.get((sid, i)) for i in idxs], "couleur_v": METHODE_COULEUR,
                         "source": f"lectures_v30_{MATCH}.json (PARSeq affiné, lecture rapide)", "reflexion": "parseq"}
    compteur["questions"] = len(a_lire)
    print(f"   LECTURE RAPIDE : {len(a_lire)} morceau(x), {len(_res)} images lues en {time.time() - t_l:.0f} s "
          f"(vidéo {t_dec:.0f} s en {_n} processus, PARSeq {time.time() - t_l - t_dec:.0f} s) — les deux équipes, réutilisable "
          f"pour tout joueur du match")
elif a_lire:
    import concurrent.futures
    client = types = None
    # v3.0 : pas de Gemini pour la lecture des numéros (PARSeq affiné)
    t_l = time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        for sid, entree, ims in ex.map(lambda c: lire(c[1], c[2]), a_lire):
            VIGN[sid] = ims
            if entree is None:
                erreurs.add(sid)
            else:
                lectures[sid] = entree
    print(f"   {len(a_lire)} segment(s) examiné(s) en {time.time() - t_l:.0f} s | questions posées : {compteur['questions']} | "
          f"évitées par la couleur : {compteur['evitees']} | non posées (plafond) : {compteur['plafond']} | erreurs : {len(erreurs)}")
# couleurs mesurées avec une ancienne méthode : remesurées sur la vidéo (aucune question Gemini)
a_remesurer = [sid for _, sid, _ in candidats if sid in lectures and sid not in VIGN
               and lectures[sid].get("couleur_v") != METHODE_COULEUR]
for sid in a_remesurer:
    ims, parts, images = vignettes(sid, lectures[sid]["images"])
    if images == lectures[sid]["images"]:
        lectures[sid]["parts"], lectures[sid]["couleur_v"] = parts, METHODE_COULEUR
        VIGN[sid] = ims
if a_remesurer:
    print(f"   couleurs remesurées (nouvelle méthode, sans Gemini) : {len(a_remesurer)} segment(s)")
sauver_lectures()
print(f"   lectures sauvegardées : {FICHIER_LECTURES} (à garder : sortie du notebook, puis dataset en entrée)")


def evaluer(sid):
    """Portions retenues d'un segment : [(image de début, image de fin)], et la raison du verdict."""
    e = lectures.get(sid)
    if not e or e.get("numeros") is None:
        return [], "plafond de questions atteint" if sid not in erreurs and e is None else "question en erreur"
    if e.get("non_lu") == "plafond":
        return [], "plafond de questions atteint (à lire au prochain run)"
    if e.get("non_lu"):
        return [], "moins de 2 vignettes de la bonne couleur (question évitée)"
    vig = sorted(zip(e["images"], e["numeros"], [equipe_mesuree(p) for p in e["parts"]]))
    if not vig:
        return [], "aucune vignette"
    pos = segments[sid]["pos"]
    portions, suites, courant = [], [], []
    for i, v in enumerate(vig):
        # vignette conforme : bonne couleur ET pas d'autre numéro lu (v2.5 : un autre numéro, même lu une seule
        # fois, arrête la portion comme une vignette de mauvaise couleur)
        if v[2] == EQ and (v[1] is None or v[1] == NUMERO):
            courant.append(i)
        elif courant:
            suites.append(courant); courant = []
    if courant:
        suites.append(courant)
    for s in suites:
        c = collections.Counter(vig[i][1] for i in s if vig[i][1] is not None)
        autre = any(n != NUMERO and k >= MIN_CONCORDANTES for n, k in c.items())
        ok = c[NUMERO] >= MIN_CONCORDANTES and not autre
        if not ok and LECTURE_UNIQUE_SI_COULEUR:
            ok = c[NUMERO] >= 1 and len(c) == 1 and len(s) >= 2
        if ok:
            # v2.6 : prolongée au plus de EXTENSION_MAX_S au-delà des vignettes lues, et seulement si aucune vignette
            # non conforme n'est au-delà ; jamais au-delà des bords du segment
            ext = EXTENSION_MAX_S * FPS
            a = max(min(pos), vig[s[0]][0] - ext) if s[0] == 0 else vig[s[0]][0]
            b = min(max(pos), vig[s[-1]][0] + ext) if s[-1] == len(vig) - 1 else vig[s[-1]][0]
            portions.append((a, b))
    lus = [v[1] for v in vig if v[1] is not None]
    if portions:
        coupe = any(v[1] is not None and v[1] != NUMERO for v in vig)
        return portions, "retenu" + (" (coupé à un autre numéro)" if coupe else "")
    if NUMERO not in lus:
        return [], "autre numéro" if lus else "aucun numéro lisible"
    if not any(v[2] == EQ for v in vig):
        return [], f"{NUMERO} lu, mais aucune vignette de la bonne couleur"
    return [], f"{NUMERO} lu moins de {MIN_CONCORDANTES} fois sur des vignettes consécutives de la bonne couleur sans autre numéro"


verdicts = {sid: evaluer(sid) for _, sid, _ in candidats}

# ══ 19/11 (5) : MODE GARDIEN — couleur de SON maillot, puis VÉRIFICATION PAR GEMINI de chaque passage ══
# Les règles de position (côté de l'image, bas de l'image, pelouse sous les pieds) ont été RETIRÉES : la vidéo suit le
# jeu (l'image se déplace), donc « gauche/droite de l'image » n'est pas un côté du terrain ; un entraîneur derrière la
# barrière se tenait sur de l'herbe, au milieu de l'image (captures du 19/11). Le vrai gardien, loin, était trop petit
# pour le seuil des numéros (90 px) : seuil propre au gardien (40 px), car aucun numéro n'est à lire.
if GARDIEN:
    _c_g = COULEUR_MAILLOT.lower()
    if not SOURCES_L:
        print("   ⚠️ MODE GARDIEN SANS FICHIER DE LECTURES en entrée (lectures_v30_<match>_v2.json) : les numéros ont été "
              "relus pour rien (≈ 1 h 30 sur processeur) et les joueurs de champ ne peuvent pas être écartés d'avance")
    _arb = AUTRES_COULEURS["arbitre"].most_common(1)[0][0] if AUTRES_COULEURS["arbitre"] else None
    _gk = AUTRES_COULEURS["gardiens"]
    print(f"\n   MODE GARDIEN : maillot « {_c_g} » | équipes {COULEUR_EQUIPE} | gardiens vus par Gemini "
          f"{dict(_gk) or '—'} | arbitre « {_arb or '?'} »")
    _risques = (["l'arbitre"] if _arb == _c_g else []) + [f"l'équipe {e}" for e, c in COULEUR_EQUIPE.items() if c == _c_g]
    _risques += [f"un autre gardien ({e})" for (c, e), k in _gk.items() if c == _c_g and e not in ("?", "AB"[EQ])]
    if _risques:
        print(f"   ⚠️ RISQUE DE CONFUSION : « {_c_g} » est aussi la couleur de {', '.join(_risques)} — la vérification "
              f"Gemini doit les écarter ; regarder la planche")

    def _part_maillot(img, b):
        """Part des pixels du haut du maillot dans la couleur du gardien, et dans la couleur d'équipe la plus présente."""
        Hh, Ww = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in b]; h, w = y2 - y1, x2 - x1
        haut = img[max(0, y1 + int(0.15 * h)):max(0, y1 + int(0.45 * h)), max(0, x1 + w // 5):max(0, x2 - w // 5)]
        if haut.size == 0:
            return 0.0, 0.0
        hv = _hsv(haut); noms = noms_pixels(hv)
        autour = _hsv(img[min(Hh, y2):min(Hh, y2 + max(4, h // 10)), max(0, x1):max(0, x2)])
        garde = np.ones(len(noms), bool)
        if len(autour):
            gz = autour[noms_pixels(autour) == "vert"]
            if len(gz) > 30:
                m = np.median(gz, axis=0)
                garde = ~((np.abs(hv[:, 0] - m[0]) <= 8) & (np.abs(hv[:, 1] - m[1]) <= 60) & (np.abs(hv[:, 2] - m[2]) <= 45))
        if garde.mean() < 0.25:
            return 0.0, 0.0
        noms = noms[garde]
        return (float(np.isin(noms, [_c_g]).mean()),
                max([float(np.isin(noms, list(nom_mesure(c))).mean()) for c in COULEUR_EQUIPE.values()] or [0.0]))

    # 1) candidats du gardien : seuil de taille propre (40 px), dans le temps de jeu
    #    19/11 (7) : on écarte d'avance les JOUEURS DE CHAMP déjà mesurés (fichier des lectures : couleur de leur
    #    équipe nette sur au moins 2 images sur 3) — sans risque pour le gardien, dont le maillot n'est pas celui de l'équipe.
    _cand_g, _champ = [], 0
    for sid, v in segments.items():
        if _FEN and not any(debut(sid) <= b_ and fin(sid) >= a_ for a_, b_ in _FEN):
            continue
        nums = sorted(v["pos"])
        if (nums[-1] - nums[0]) / FPS < MIN_DUREE_S:
            continue
        e_l = lectures.get(sid) or {}
        if e_l.get("parts") and e_l.get("couleur_v") == METHODE_COULEUR:
            nets = sum(1 for p_ in e_l["parts"] if max(p_) >= SEUIL_COULEUR)
            if nets >= max(2, (2 * len(e_l["parts"]) + 2) // 3):
                _champ += 1
                continue
        ok_h = [n for n in nums if v["pos"][n][0][3] - v["pos"][n][0][1] >= HAUTEUR_MIN_GARDIEN]
        if len(ok_h) >= 2:
            _cand_g.append((sid, list(dict.fromkeys(ok_h[int(round(q * (len(ok_h) - 1)))]
                                                  for q in np.linspace(0, 1, min(IMAGES_PAR_PASSAGE_GARDIEN, len(ok_h)))))))
    print(f"   segments examinés pour le gardien : {len(_cand_g)} (≥ {MIN_DUREE_S} s, ≥ 2 images de {HAUTEUR_MIN_GARDIEN} px) "
          f"| joueurs de champ écartés d'avance (couleur de leur équipe déjà mesurée) : {_champ}")

    # 2) couleur du maillot — 19/11 (7) : vidéo lue D'UN SEUL TRAIT en 2 processus (comme la lecture des numéros)
    def _lire_tranche_gardien(taches):
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(VIDEO); fps_v = cap.get(cv2.CAP_PROP_FPS) or FPS
        out, cur = [], None
        for num, items in taches:
            cible = int(round((KO + num / FPS) * fps_v))
            if cur is None or cible < cur or cible - cur > 300:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cible); cur = cible
            while cur < cible:
                cap.grab(); cur += 1
            ok, img = cap.read(); cur += 1
            if ok:
                for sid, b in items:
                    out.append((sid, int(num), _part_maillot(img, b)))
        cap.release()
        return out

    import multiprocessing as _mp_g
    t_g = time.time()
    _besoin_g = collections.defaultdict(list)
    for sid, ch in _cand_g:
        for num in ch:
            _besoin_g[int(num)].append((sid, segments[sid]["pos"][num][0]))
    _taches_g = sorted(_besoin_g.items()); _ng = max(1, N_PROCESSUS_LECTURE)
    _morc_g = [_taches_g[i * len(_taches_g) // _ng:(i + 1) * len(_taches_g) // _ng] for i in range(_ng)]
    with _mp_g.get_context("fork").Pool(_ng) as _pool_g:
        _res_g = [r for part in _pool_g.map(_lire_tranche_gardien, _morc_g) for r in part]
    _noirs_tmp = collections.defaultdict(list)
    for sid, num, (pg, pe) in _res_g:
        if pg >= SEUIL_COULEUR and pg >= 1.5 * pe:
            _noirs_tmp[sid].append(num)
    _noirs = {sid: sorted(v) for sid, v in _noirs_tmp.items() if len(v) >= 2}
    print(f"   couleur mesurée sur {len(_res_g)} image(s) en {time.time() - t_g:.0f} s ({_ng} processus) | "
          f"segments en « {_c_g} » : {len(_noirs)}")
    # 3) VÉRIFICATION PAR GEMINI, ÉCONOME (19/11 (6)) :
    #    • un passage par PERSONNE suivie : les passages de la même piste du suivi (même tid), à moins de 5 min l'un
    #      de l'autre, sont la même personne → une seule vérification pour le groupe ;
    #    • PLUSIEURS personnes par question : une planche de VIGNETTES_PAR_QUESTION images (chaque image = la personne
    #      encadrée en magenta, avec ce qui l'entoure : barrière, banc, but…).
    from google import genai as _gg3
    from google.genai import types as _gt3
    _cl3 = _gg3.Client(api_key=GEMINI_KEY); _jet3 = {"q": 0, "e": 0, "s": 0}
    _groupes = collections.defaultdict(list)
    for sid in sorted(_noirs, key=debut):
        tid = segments[sid]["tid"]; g = _groupes.get((tid, 0))
        k = 0
        while (tid, k) in _groupes and debut(sid) - fin(_groupes[(tid, k)][-1]) > 300:
            k += 1
        _groupes[(tid, k)].append(sid)
    _reps = sorted(_groupes.items(), key=lambda kv: -sum(len(_noirs[s]) for s in kv[1]))[:MAX_VERIF_GARDIEN]
    print(f"   passages en « {_c_g} » regroupés par personne suivie : {len(_noirs)} passages → {len(_groupes)} personne(s)-"
          f"période(s) | vérifiées : {len(_reps)} | {VIGNETTES_PAR_QUESTION} par question → "
          f"{(len(_reps) + VIGNETTES_PAR_QUESTION - 1) // VIGNETTES_PAR_QUESTION} question(s)")

    def _tuile(sid):
        ims = _noirs[sid]; num = ims[len(ims) // 2]
        ok, img = image_video(num)
        if not ok:
            return None
        x1, y1, x2, y2 = [int(v) for v in segments[sid]["pos"][num][0]]; h = max(20, y2 - y1)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        X1, X2 = max(0, cx - 3 * h), min(img.shape[1], cx + 3 * h); Y1, Y2 = max(0, cy - 2 * h), min(img.shape[0], cy + 2 * h)
        im2 = img.copy(); cv2.rectangle(im2, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 0, 255), 2)
        return cv2.resize(im2[Y1:Y2, X1:X2], (360, 240))

    _Q3 = (f"Football match. Each numbered tile shows ONE person inside a MAGENTA box, with what surrounds him. For each "
           f"tile, is this person the GOALKEEPER currently playing on the pitch (shirt « {_c_g} »), or someone else: coach, "
           f"substitute, staff, spectator, ball boy, referee, player of another color? A goalkeeper is ON the pitch, "
           f"usually near his goal; people behind fences or barriers, or along the touchline outside the field, are NOT "
           f"the goalkeeper. Answer ONLY with JSON: {{\"gardien\": [true/false per tile, in order], "
           f"\"qui\": [short description per tile]}}")
    _verif = {}
    for i0 in range(0, len(_reps), VIGNETTES_PAR_QUESTION):
        lot = [(cle, sids, _tuile(max(sids, key=lambda s: len(_noirs[s])))) for cle, sids in _reps[i0:i0 + VIGNETTES_PAR_QUESTION]]
        lot = [x for x in lot if x[2] is not None]
        if not lot:
            continue
        tuiles = []
        for j, (_, _, t) in enumerate(lot, 1):
            t = t.copy(); cv2.putText(t, str(j), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3); tuiles.append(t)
        while len(tuiles) % 3:
            tuiles.append(np.zeros_like(tuiles[0]))
        planche = cv2.vconcat([cv2.hconcat(tuiles[k:k + 3]) for k in range(0, len(tuiles), 3)])
        try:
            r = _cl3.models.generate_content(model=MODELE, contents=[_Q3, _gt3.Part.from_bytes(
                data=cv2.imencode(".jpg", planche, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), mime_type="image/jpeg")],
                config={"temperature": 0.0, "thinking_config": {"thinking_level": "low"}})
            um = getattr(r, "usage_metadata", None); _jet3["q"] += 1
            _jet3["e"] += int(getattr(um, "prompt_token_count", 0) or 0)
            _jet3["s"] += int(getattr(um, "candidates_token_count", 0) or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
            txt = r.text or ""; d, _ = json.JSONDecoder().raw_decode(txt[txt.index("{"):])
            gk, qui = list(d.get("gardien") or []), list(d.get("qui") or [])
            for j, (cle, sids, _) in enumerate(lot):
                dec = (bool(gk[j]) if j < len(gk) else False, str(qui[j] if j < len(qui) else "")[:60])
                for sid in sids:
                    _verif[sid] = dec
        except Exception as ex:
            for cle, sids, _ in lot:
                for sid in sids:
                    _verif[sid] = (False, f"réponse inexploitable ({str(ex)[:40]})")
    _cout3 = _jet3["e"] / 1e6 * 0.75 + _jet3["s"] / 1e6 * 3.75
    print(f"   VÉRIFICATION GEMINI : {_jet3['q']} question(s) | passages jugés gardien : "
          f"{sum(1 for v in _verif.values() if v[0])} | autre : {sum(1 for v in _verif.values() if not v[0])} | ≈ {_cout3:.3f} $")
    _qui = collections.Counter(v[1].lower() for v in _verif.values() if not v[0])
    if _qui:
        print("   écartés par Gemini (exemples) : " + " ; ".join(f"{q} ×{n}" for q, n in _qui.most_common(6)))
    # 4) verdicts (+ fiches de lecture minimales pour la planche et les raccords)
    verdicts, _raisons_g = {sid: v for sid, v in verdicts.items() if sid not in _noirs}, collections.Counter()
    for sid in list(verdicts):
        verdicts[sid] = ([], f"pas de maillot {_c_g}")
    for sid, ims in _noirs.items():
        g = _verif.get(sid)
        if g is None:
            verdicts[sid] = ([], "maillot noir, non vérifié (plafond)")
        elif g[0]:
            verdicts[sid] = ([(min(ims), max(ims))], f"retenu (maillot {_c_g}, gardien selon Gemini)")
        else:
            verdicts[sid] = ([], f"maillot {_c_g}, mais pas le gardien selon Gemini ({g[1]})")
        if verdicts[sid][0] and sid not in lectures:
            lectures[sid] = {"images": ims, "numeros": [None] * len(ims), "parts": [[0.0, 0.0]] * len(ims),
                             "choix": ims, "source": "mode gardien"}
        _raisons_g[verdicts[sid][1].split(" (")[0] if not verdicts[sid][0] else "retenu"] += 1
    print(f"   MODE GARDIEN : {dict(_raisons_g)}")

# ── v3.2 : CONFIRMATION GEMINI des segments où le numéro n'a été lu qu'une fois ──
CONFIRMATION_GEMINI, MAX_CONFIRMATIONS = (not GARDIEN), 80
_a_conf = [sid for sid, (p, r) in verdicts.items() if not p and str(r).startswith(f"{NUMERO} lu moins")]
if CONFIRMATION_GEMINI and _a_conf and GEMINI_KEY:
    from google import genai as _gg
    from google.genai import types as _gt
    _cl = _gg.Client(api_key=GEMINI_KEY); _jet = {"q": 0, "e": 0, "s": 0}; _gagnes = 0
    _Q = """These images show (normally) the same amateur football player at several moments. For EACH image, give the
shirt number ONLY if its digits are clearly readable (back, front or shorts); otherwise null. Do not guess.
Answer ONLY with JSON: {"numbers": [one value per image, in order]}"""
    for sid in _a_conf[:MAX_CONFIRMATIONS]:   # (12 images par morceau : la planche envoyée à Gemini est plus large)
        e = lectures[sid]
        try:
            ims, _, _ = vignettes(sid, e["choix"])
            h = 200; pl = cv2.hconcat([cv2.resize(i, (max(8, int(i.shape[1] * h / i.shape[0])), h)) for i in ims])
            r = _cl.models.generate_content(model=MODELE, contents=[_Q, _gt.Part.from_bytes(
                data=cv2.imencode(".jpg", pl, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), mime_type="image/jpeg")],
                config={"temperature": 0.0, "thinking_config": {"thinking_level": "low"}})
            um = getattr(r, "usage_metadata", None); _jet["q"] += 1
            _jet["e"] += int(getattr(um, "prompt_token_count", 0) or 0)
            _jet["s"] += int(getattr(um, "candidates_token_count", 0) or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
            txt = r.text or ""; d, _ = json.JSONDecoder().raw_decode(txt[txt.index("{"):])
            gem = [int(x) if str(x).isdigit() else None for x in (d.get("numbers") or [])][:len(e["numeros"])]
            gem += [None] * (len(e["numeros"]) - len(gem))
            e["numeros_parseq"] = list(e["numeros"])
            e["numeros"] = [g if g is not None else p for g, p in zip(gem, e["numeros"])]   # Gemini prime s'il lit
            e["source"] = str(e.get("source")) + " + confirmation Gemini"
            verdicts[sid] = evaluer(sid); _gagnes += bool(verdicts[sid][0])
        except Exception as ex:
            print(f"   ⚠️ confirmation {sid} : {ex}")
    print(f"   CONFIRMATION GEMINI : {len(_a_conf)} segment(s) lus une seule fois | {_jet['q']} relu(s) | "
          f"{_gagnes} désormais retenu(s) | ≈ {_jet['e'] / 1e6 * 0.75 + _jet['s'] / 1e6 * 3.75:.3f} $")
# ── v3.2 : DIAGNOSTIC D'UNE FENÊTRE (gratuit) : où la chaîne casse-t-elle pour ce joueur ? ──
DIAG_FENETRES = [] if BOUT_EN_BOUT else [("81:50", "82:02"), ("87:15", "87:30")]   # [] = pas de diagnostic (n°16 vu à 81:56 et 87:21)
for DIAG_FENETRE in DIAG_FENETRES:
    _s = lambda h: int(h.split(":")[0]) * 60 + int(h.split(":")[1])
    _a, _b = _s(DIAG_FENETRE[0]), _s(DIAG_FENETRE[1])
    _cand = {sid: ch for _, sid, ch in candidats}
    print(f"\n   DIAGNOSTIC {DIAG_FENETRE[0]}-{DIAG_FENETRE[1]} : morceaux du suivi qui touchent la fenêtre")
    _lignes = []
    for sid, v in sorted(segments.items(), key=lambda x: debut(x[0])):
        if not (debut(sid) <= _b and fin(sid) >= _a):
            continue
        pos = v["pos"]; nums = sorted(pos)
        dur = (nums[-1] - nums[0]) / FPS
        h_med = float(np.median([pos[n][0][3] - pos[n][0][1] for n in nums]))
        part_eq = sum(1 for _, e in pos.values() if e == EQ) / len(pos)
        if sid in _cand:
            e = lectures.get(sid) or {}
            raison = verdicts.get(sid, ([], "?"))[1]
            detail = f"lu : {e.get('numeros')} | couleurs : {[equipe_mesuree(p) for p in e.get('parts') or []]} | {raison}"
            choix = _cand[sid]
        else:
            lis = [n for n in nums if pos[n][0][3] - pos[n][0][1] >= MIN_HAUTEUR]
            why = ("trop court (< %.1f s)" % MIN_DUREE_S if dur < MIN_DUREE_S else
                   "moins de 2 images assez grandes" if len(lis) < 2 else
                   f"part de l'équipe {part_eq:.0%} < {PREFILTRE_EQUIPE:.0%}" if part_eq < PREFILTRE_EQUIPE else "non retenu (plafond ?)")
            detail = f"NON EXAMINÉ : {why}"
            choix = [nums[int(round(q * (len(nums) - 1)))] for q in np.linspace(0, 1, min(4, len(nums)))]
        print(f"     {sid:24s} {fmt(debut(sid))}-{fmt(fin(sid))} ({dur:4.1f} s, hauteur ~{h_med:.0f} px, équipe {part_eq:.0%}) | {detail}")
        try:
            ims, _, _ = vignettes(sid, choix)
            if ims:
                hh = 220; l = cv2.hconcat([cv2.resize(i, (max(8, int(i.shape[1] * hh / i.shape[0])), hh)) for i in ims])
                cv2.putText(l, f"{fmt(debut(sid))} {sid[:14]}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                _lignes.append(l)
        except Exception:
            pass
    if _lignes:                                    # 07/11 : pages de 12 lignes (une seule image trop haute était réduite à l'envoi)
        _w = max(l.shape[1] for l in _lignes)
        for _k in range(0, len(_lignes), 12):
            _nomf = f"{SORTIE}/diagnostic_fenetre_{NUMERO}_{DIAG_FENETRE[0].replace(':', '-')}_p{_k // 12 + 1}.jpg"
            cv2.imwrite(_nomf, np.vstack([np.pad(l, ((0, 4), (0, _w - l.shape[1]), (0, 0))) for l in _lignes[_k:_k + 12]]))
        print(f"     planches : diagnostic_fenetre_{NUMERO}_{DIAG_FENETRE[0].replace(':', '-')}_p1…p{(len(_lignes) - 1) // 12 + 1}.jpg "
              f"(12 morceaux par page, dans l'ordre ci-dessus ; lignes 1-12 en p1, 13-24 en p2…)")

retenus = [sid for sid, (p, _) in verdicts.items() if p]
rejets = collections.Counter(r for p, r in verdicts.values() if not p)
print(f"   RETENUS (numéro lu) pour le {NUMERO} {COULEUR} : {len(retenus)} segment(s) | écartés : {dict(rejets)}")

# planche des lectures
nom_eq = lambda q: COULEUR_EQUIPE.get(q, "?") if q is not None else "?"
douteux = [sid for _, sid, _ in candidats if sid not in retenus and NUMERO in ((lectures.get(sid) or {}).get("numeros") or [])]
lignes_v = []


import unicodedata
ascii_ = lambda t: unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()   # OpenCV : pas d'accents


def ligne_planche(titre, coul, cases):
    titre = ascii_(titre)
    l = cv2.hconcat(cases)
    l = cv2.copyMakeBorder(l, 40, 0, 0, max(0, 760 - l.shape[1]), cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(l, titre[:95], (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, coul, 1)
    return l[:, :760]


def case(im, texte, bon):
    c = cv2.copyMakeBorder(cv2.resize(im, (120, 256)), 0, 26, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(c, ascii_(texte), (4, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0) if bon else (0, 0, 255), 1)
    return c


def ecrire_planches(lignes, base):
    """Pages de LIGNES_PAR_PAGE lignes : base_p1.jpg, base_p2.jpg… Renvoie la liste des fichiers écrits."""
    for ancien in glob.glob(f"{base}_p*.jpg"):
        os.remove(ancien)
    pages = [lignes[k:k + LIGNES_PAR_PAGE] for k in range(0, len(lignes), LIGNES_PAR_PAGE)]
    noms = []
    for i, p in enumerate(pages, 1):
        noms.append(f"{base}_p{i}.jpg"); cv2.imwrite(noms[-1], cv2.vconcat(p))
    return noms


for sid, etiquette, coul in [(s, "RETENU", (0, 200, 0)) for s in sorted(retenus, key=debut)] + \
                           [(s, "ECARTE", (0, 0, 255)) for s in sorted(douteux, key=debut)]:
    e = lectures[sid]
    ims = VIGN.get(sid) or vignettes(sid, e["images"])[0]
    if not ims:
        continue
    cases = [case(im, f"{n if n is not None else '-'} {nom_eq(equipe_mesuree(p))}", equipe_mesuree(p) == EQ)
             for im, n, p in zip(ims, e["numeros"], e["parts"])]
    portions = verdicts[sid][0]
    txt = f"{etiquette} | {sid} | {fmt(debut(sid))}-{fmt(fin(sid))} | " + (
        "garde " + ", ".join(f"{fmt(KO + a / FPS)}-{fmt(KO + b / FPS)}" for a, b in portions) if portions else verdicts[sid][1])
    lignes_v.append(ligne_planche(txt, coul, cases))
if lignes_v:
    pages = ecrire_planches(lignes_v, f"{SORTIE}/verification_lectures_v30")
    print(f"   planche des lectures : {len(pages)} page(s), {len(retenus)} retenus puis {len(douteux)} écartés où le "
          f"{NUMERO} est lu ({os.path.basename(pages[0])}…)")
with open(f"{SORTIE}/segments_v28_{MATCH}_{NUMERO}_{COULEUR}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["segment", "piste", "debut", "fin", "coupe_avant", "images_lues", "numeros", "couleurs_mesurees", "verdict", "portions"])
    for _, sid, _ in sorted(candidats, key=lambda c: debut(c[1])):
        e = lectures.get(sid) or {}
        w.writerow([sid, segments[sid]["tid"], fmt(debut(sid)), fmt(fin(sid)), segments[sid]["raison"] or "",
                    " ".join(fmt(KO + n / FPS) for n in e.get("images", [])),
                    " ".join("-" if n is None else str(n) for n in (e.get("numeros") or [])),
                    " ".join(nom_eq(equipe_mesuree(p)) for p in e.get("parts", [])),
                    verdicts[sid][1], " ".join(f"{fmt(KO + a / FPS)}-{fmt(KO + b / FPS)}" for a, b in verdicts[sid][0])])
if not retenus:
    _fiche_joueur(False)
    _ecrire_resultat(trouve=False, montage=None, fiche=f"{SORTIE}/fiche_joueur_{MATCH}_{NUMERO}.pdf",
                     rapport_et_fiche=(f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf"
                                       if os.path.exists(f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf") else None),
                     message=f"Nous n'avons trouvé aucune action du joueur n°{NUMERO} ({COULEUR}) dans cette vidéo. "
                             f"Vérifiez le numéro et la couleur du maillot indiqués. Le résumé du match vous est livré normalement.")
    raise SystemExit(f"⚠️ Aucune action trouvée pour le n°{NUMERO} ({COULEUR}) : vérifiez le numéro et la couleur du maillot. "
                     f"Le rapport du match reste valable ; la fiche du joueur l'indique (fiche_joueur_{MATCH}_{NUMERO}.pdf).")


# ── 4 bis. RACCORDEMENT DES SEGMENTS VOISINS (sans Gemini) ────────────────────────────────────
nums_seg = {sid: sorted(v["pos"]) for sid, v in segments.items()}
par_debut = sorted((n[0], sid) for sid, n in nums_seg.items())
par_fin = sorted((n[-1], sid) for sid, n in nums_seg.items())
_cles_debut = [d for d, _ in par_debut]
_cles_fin = [f for f, _ in par_fin]


def _bas(b):
    return (b[0] + b[2]) / 2, b[3], max(1.0, b[3] - b[1])


def distance_h(b1, b2):
    x1, y1, h1 = _bas(b1); x2, y2, h2 = _bas(b2)
    return float(np.hypot(x1 - x2, y1 - y2) / ((h1 + h2) / 2))


def couleurs_segment(sid):
    """[(image, équipe mesurée)] dans l'ordre du temps : lectures existantes, sinon 3 vignettes (début, milieu, fin)."""
    e = lectures.get(sid)
    if e and e.get("couleur_v") == METHODE_COULEUR and e.get("parts"):
        return sorted((n, equipe_mesuree(p)) for n, p in zip(e["images"], e["parts"]))
    if sid in COUL_R and COUL_R[sid].get("couleur_v") == METHODE_COULEUR:
        return sorted((n, equipe_mesuree(p)) for n, p in zip(COUL_R[sid]["images"], COUL_R[sid]["parts"]))
    pos, nums = segments[sid]["pos"], nums_seg[sid]
    grands = [n for n in nums if pos[n][0][3] - pos[n][0][1] >= RACCORD_HMIN] or nums
    choix = list(dict.fromkeys([grands[0], grands[len(grands) // 2], grands[-1]]))
    images, parts = [], []
    for n in choix:
        ok, img = image_video(n)
        if ok:
            images.append(int(n)); parts.append(parts_couleur(img, pos[n][0]))
    COUL_R[sid] = {"images": images, "parts": parts, "couleur_v": METHODE_COULEUR}
    return sorted((n, equipe_mesuree(p)) for n, p in zip(images, parts))


def contredit(sid):
    """Un autre numéro a-t-il été lu sur une vignette de la bonne couleur de ce segment ?"""
    e = lectures.get(sid) or {}
    return any(n is not None and n != NUMERO and equipe_mesuree(p) == EQ
               for n, p in zip(e.get("numeros") or [], e.get("parts") or []))


def voisins(sid_a, bord, sens):
    """Segments plausibles au raccord : (segment, écart en s, distance en hauteurs, image du raccord)."""
    bb_a = segments[sid_a]["pos"][bord][0]
    out = []
    if sens > 0:
        i0 = bisect.bisect_left(_cles_debut, bord - RACCORD_RECOUV_S * FPS)
        i1 = bisect.bisect_right(_cles_debut, bord + RACCORD_MAX_S * FPS)
        cands = [sid for _, sid in par_debut[i0:i1] if sid != sid_a and nums_seg[sid][-1] > bord]
    else:
        i0 = bisect.bisect_left(_cles_fin, bord - RACCORD_MAX_S * FPS)
        i1 = bisect.bisect_right(_cles_fin, bord + RACCORD_RECOUV_S * FPS)
        cands = [sid for _, sid in par_fin[i0:i1] if sid != sid_a and nums_seg[sid][0] < bord]
    pos_a = segments[sid_a]["pos"]
    for sid in cands:
        nums, pos = nums_seg[sid], segments[sid]["pos"]
        j = nums[0] if sens > 0 else nums[-1]
        chevauche = (sens > 0 and j <= bord) or (sens < 0 and j >= bord)
        if chevauche:                                                   # chevauchement : comparer à la même image
            communs = [n for n in nums if n in pos_a]
            if communs:
                n = min(communs, key=lambda k: abs(k - bord))
                d, ecart = distance_h(pos_a[n][0], pos[n][0]), 0.0
            else:
                n = min(nums, key=lambda k: abs(k - bord))
                d, ecart = distance_h(bb_a, pos[n][0]), 0.0
        else:
            d, ecart = distance_h(bb_a, pos[j][0]), abs(j - bord) / FPS
        if d <= RACCORD_BASE_H + RACCORD_H_PAR_S * ecart:
            out.append((sid, round(ecart, 2), round(d, 2), j, chevauche))
    return out


def statut_couleur(cs):
    n_eq = sum(1 for _, q in cs if q == EQ)
    n_autre = sum(1 for _, q in cs if q is not None and q != EQ)
    if n_autre > n_eq:
        return "autre équipe"
    return "bonne couleur" if n_eq >= 2 and n_autre == 0 else "incertaine"


VARIANTES = {"ref": (False, True), "B": (True, True), "B partiel (v2.4)": (True, False)}   # (plus proche, entier seul)
ABREGE = {"ref": "R", "B": "B", "B partiel (v2.4)": "Bp"}
if RACCORD_MONTAGE not in ("ref", "B"):
    raise SystemExit(f"❌ RACCORD_MONTAGE = {RACCORD_MONTAGE!r} : choisis « ref » ou « B »")
_infos_cache = {}


def infos_voisins(sid_a, bord, sens):
    """Voisins plausibles avec leur couleur : (segment, écart s, distance h, couleurs, statut, nb positions).
    Mis en cache : les quatre variantes partagent exactement les mêmes mesures."""
    cle = (sid_a, bord, sens)
    if cle not in _infos_cache:
        brut = voisins(sid_a, bord, sens)
        # v2.6 : deux pistes présentes au même moment et éloignées = deux joueurs différents (pas de couleur mesurée)
        _infos_cache[cle] = [(sid, ecart, d, [] if (chev and d > RACCORD_RECOUV_MAX_H) else couleurs_segment(sid),
                              "chevauchement éloigné" if (chev and d > RACCORD_RECOUV_MAX_H) else None,
                              len(nums_seg[sid])) for sid, ecart, d, j, chev in brut]
        _infos_cache[cle] = [(i[0], i[1], i[2], i[3], i[4] or statut_couleur(i[3]), i[5]) for i in _infos_cache[cle]]
    return _infos_cache[cle]


def raccorder(nom_variante):
    """Raccords d'une variante. Renvoie (raccords [(segment, début, fin)], trace [une ligne par décision])."""
    plus_proche, entier_seul = VARIANTES[nom_variante]
    raccords, trace = [], []
    couverts = {sid: list(p) for sid, (p, _) in verdicts.items() if p}

    def chercher(sid_a, bord, sens, profondeur, origine):
        if profondeur > RACCORD_MAX_CHAINE:
            return
        infos = infos_voisins(sid_a, bord, sens)
        concurrents = [i for i in infos if i[4] not in ("autre équipe", "chevauchement éloigné")]
        ligne = {"variante": nom_variante, "origine": origine, "depuis": sid_a, "sens": "après" if sens > 0 else "avant",
                 "profondeur": profondeur, "bord": fmt(KO + bord / FPS),
                 "voisins": " ".join(f"{i[0]}({i[4]},{i[5]}pos,{i[2]}h,{i[1]}s)" for i in infos)}
        if not concurrents:
            trace.append({**ligne, "decision": "aucun voisin plausible"}); return
        regle = ""
        if len(concurrents) > 1:
            tri = sorted(concurrents, key=lambda i: i[2])
            if plus_proche and tri[1][2] >= RACCORD_RAPPORT * max(tri[0][2], 0.05):   # B
                concurrents, regle = [tri[0]], f" [B : {tri[0][2]} h contre {tri[1][2]} h]"
            else:
                trace.append({**ligne, "decision": f"ambigu ({len(concurrents)} candidats non exclus par la couleur)"}); return
        sid, ecart, d, cs, statut, _ = concurrents[0]
        ligne.update({"candidat": sid, "ecart_s": ecart, "distance_h": d, "couleurs": " ".join(nom_eq(q) for _, q in cs)})
        if sid in couverts:
            trace.append({**ligne, "decision": "déjà retenu (lien seulement)" + regle}); return
        if statut != "bonne couleur":
            trace.append({**ligne, "decision": f"refusé : couleur {statut}" + regle}); return
        if contredit(sid):
            trace.append({**ligne, "decision": "refusé : autre numéro lu sur ce segment" + regle}); return
        nums = nums_seg[sid]
        ordre = cs if sens > 0 else cs[::-1]                 # du côté du raccord vers l'autre bout
        bons = 0
        while bons < len(ordre) and ordre[bons][1] == EQ:
            bons += 1
        if bons == 0:
            trace.append({**ligne, "decision": "refusé : mauvaise couleur au raccord" + regle}); return
        entier = bons == len(ordre)
        if entier_seul and not entier:
            trace.append({**ligne, "decision": "refusé : couleur incomplète (raccord en partie)" + regle}); return
        if sens > 0:
            a, b = nums[0], (nums[-1] if entier else ordre[bons - 1][0])
        else:
            a, b = (nums[0] if entier else ordre[bons - 1][0]), nums[-1]
        raccords.append((sid, a, b)); couverts[sid] = [(a, b)]
        trace.append({**ligne, "decision": "ACCEPTÉ" + ("" if entier else " (en partie)") + regle,
                      "portion": f"{fmt(KO + a / FPS)}-{fmt(KO + b / FPS)}"})
        if entier:
            chercher(sid, nums[-1] if sens > 0 else nums[0], sens, profondeur + 1, origine)

    for sid in sorted(retenus, key=debut):
        for a, b in verdicts[sid][0]:
            nums = nums_seg[sid]
            if b == nums[-1]:          # la portion va jusqu'au bout du segment : on peut chercher la suite
                chercher(sid, b, +1, 1, sid)
            if a == nums[0]:
                chercher(sid, a, -1, 1, sid)
    return raccords, trace


t_r = time.time()
RESULTATS = {v: raccorder(v) for v in VARIANTES}
sauver_lectures()                  # couleurs mesurées pour les raccords conservées pour le prochain run
cat = lambda d: "accepté" if d.startswith("ACCEPTÉ") else "ambigu" if d.startswith("ambigu") else \
    "aucun voisin" if d.startswith("aucun") else "déjà retenu" if d.startswith("déjà") else "refusé"
print(f"\n   RACCORDS (sans Gemini, {time.time() - t_r:.0f} s) — comparaison des variantes sur les mêmes lectures :")
print(f"   {'variante':<8} {'acceptés':>9} {'ambigus':>8} {'refusés':>8} {'aucun':>6} {'déjà ret.':>9} {'durée ajoutée':>14}")
COMPARAISON = []
for v, (rac, tr) in RESULTATS.items():
    c = collections.Counter(cat(t["decision"]) for t in tr)
    duree_aj = sum((b - a) / FPS for _, a, b in rac)
    COMPARAISON.append([v, len(rac), c["ambigu"], c["refusé"], c["aucun voisin"], c["déjà retenu"], round(duree_aj, 1)])
    print(f"   {v:<8} {len(rac):>9} {c['ambigu']:>8} {c['refusé']:>8} {c['aucun voisin']:>6} {c['déjà retenu']:>9} {duree_aj:>12.0f} s"
          + ("   ← montage" if v == RACCORD_MONTAGE else ""))
with open(f"{SORTIE}/raccords_v28_{MATCH}_{NUMERO}_{COULEUR}.csv", "w", newline="", encoding="utf-8") as f:
    champs = ["variante", "origine", "depuis", "sens", "profondeur", "bord", "candidat", "ecart_s", "distance_h", "couleurs",
              "decision", "portion", "voisins"]
    w = csv.DictWriter(f, fieldnames=champs, delimiter=";", extrasaction="ignore")
    w.writeheader()
    for v, (_, tr) in RESULTATS.items():
        w.writerows(tr)
with open(f"{SORTIE}/raccords_comparaison_v28_{MATCH}_{NUMERO}_{COULEUR}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["variante", "acceptes", "ambigus", "refuses", "aucun_voisin", "deja_retenus", "duree_ajoutee_s"])
    w.writerows(COMPARAISON)
# segment raccordé -> variantes qui l'acceptent (affiché sur la planche et dans le montage)
VAR_PAR_SEG = collections.defaultdict(list)
for v, (rac, _) in RESULTATS.items():
    for sid, _, _ in rac:
        VAR_PAR_SEG[sid].append(v)
abrege = lambda vs: ",".join(ABREGE[v] for v in vs)
# planche : chaque raccord accepté par au moins une variante, une seule fois, avec les variantes qui l'acceptent
lignes_r, vus = [], set()
for v in VARIANTES:
    for t in [t for t in RESULTATS[v][1] if t["decision"].startswith("ACCEPTÉ")]:
        if t["candidat"] in vus:
            continue
        vus.add(t["candidat"])
        sid_a, sid = t["depuis"], t["candidat"]
        n_a = nums_seg[sid_a][-1] if t["sens"] == "après" else nums_seg[sid_a][0]
        cases = []
        ok, img = image_video(n_a)
        if ok and decoupe(img, segments[sid_a]["pos"][n_a][0]) is not None:
            cases.append(case(decoupe(img, segments[sid_a]["pos"][n_a][0]), "origine", True))
        for n, q in couleurs_segment(sid):
            ok, img = image_video(n)
            if ok and decoupe(img, segments[sid]["pos"][n][0]) is not None:
                cases.append(case(decoupe(img, segments[sid]["pos"][n][0]), f"{fmt(KO + n / FPS)} {nom_eq(q)}", q == EQ))
        if cases:
            lignes_r.append(ligne_planche(f"[{abrege(VAR_PAR_SEG[sid])}] {t['sens']} | {sid_a} -> {sid} | {t['ecart_s']} s, "
                                          f"{t['distance_h']} h | {t.get('portion', '')}", (255, 200, 0), cases))
if lignes_r:
    pages = ecrire_planches(lignes_r, f"{SORTIE}/verification_raccords_v28")
    print(f"   planche des raccords : {len(pages)} page(s) ([variantes qui l'acceptent : R = ref, B, Bp = B partiel v2.4], "
          f"1re vignette = origine)")
raccords = RESULTATS[RACCORD_MONTAGE][0]

# ── 5. MONTAGE AVEC REPÈRE (portions lues en jaune, raccords en bleu avec leurs variantes) ──────
positions, spans = {}, []
for sid in retenus:
    for a, b in verdicts[sid][0]:
        spans.append((KO + a / FPS, KO + b / FPS))
        for num, (bb, _) in segments[sid]["pos"].items():
            if a <= num <= b:
                positions.setdefault(num, (bb, "lu"))
for sid, a, b in raccords:
    spans.append((KO + a / FPS, KO + b / FPS))
    etiquette = ",".join(ABREGE[v] for v in VAR_PAR_SEG[sid] if v != "B partiel (v2.4)")   # ex. « R,B »
    for num, (bb, _) in segments[sid]["pos"].items():
        if a <= num <= b and positions.get(num, (None, ""))[1] != "lu":
            positions[num] = (bb, "raccord:" + etiquette)

# ── v3.1 : TOUCHES DE BALLE du joueur, d'après les événements du suivi ─────────────────────────────
_tids_lus = collections.defaultdict(set)          # image → pistes du joueur sur une portion où son numéro est lu
for sid in retenus:
    _tid = int(sid.split("@")[0])
    for a, b in verdicts[sid][0]:
        for num in segments[sid]["pos"]:
            if a <= num <= b:
                _tids_lus[num].add(_tid)
_cles_ev = [k for k in ("tracker_id", "track_id", "player_track_id", "tid") if events and isinstance(events[0], dict)
            and any(k in e for e in events[:2000])]
print(f"\n   TOUCHES : champs de piste trouvés dans les événements : {_cles_ev} | exemple : "
      f"{ {k: events[0][k] for k in list(events[0])[:10]} if events and isinstance(events[0], dict) else type(events[0]) }")
TOUCHES = []
_pas = 5                                           # le suivi garde une image sur 5
_ev_ok = [e for e in (events if isinstance(events, list) else []) if isinstance(e, dict)
          and str(e.get("type", "")) in TYPES_TOUCHES and e.get("frame") is not None]
_cles_pos = np.array(sorted(positions)) if positions else np.array([], dtype=int)


def _bb_proche(n):
    if not len(_cles_pos):
        return None
    i = int(np.searchsorted(_cles_pos, n))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(_cles_pos) and abs(int(_cles_pos[j]) - n) <= _pas:
            best = positions[int(_cles_pos[j])][0]
    return best


def _touches(decal, echelle):
    out = []
    for e in _ev_ok:
        n = int(e["frame"]) + decal
        if _cles_ev:
            pistes = _tids_lus.get(n) or set()
            if pistes and any(e.get(k) is not None and int(e.get(k)) in pistes for k in _cles_ev):
                out.append((KO + n / FPS, str(e.get("type"))))
            continue
        if e.get("x") is None or e.get("y") is None:
            continue
        bb = _bb_proche(n)
        if bb is None:
            continue
        x, y = float(e["x"]) * echelle, float(e["y"]) * echelle
        x1, y1, x2, y2 = bb; mx, my = 0.3 * (x2 - x1), 0.3 * (y2 - y1)
        if x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my:
            out.append((KO + n / FPS, str(e.get("type"))))
    return out


# 08/11 : la numérotation des images et l'échelle des coordonnées des événements ne sont pas documentées → on essaie
# les combinaisons plausibles et on garde celle qui rattache le plus d'événements (affichée pour contrôle).
_essais = {}
for _d in (0, -int(round(KO * FPS))):
    for _e in (1.0, 1920 / 853):
        _essais[(_d, _e)] = _touches(_d, _e)
(_dbest, _ebest), TOUCHES = max(_essais.items(), key=lambda kv: len(kv[1]))
if _ev_ok:
    print(f"   TOUCHES : {len(_ev_ok)} événement(s) de jeu | images du suivi {int(_cles_pos.min()) if len(_cles_pos) else '-'}"
          f"-{int(_cles_pos.max()) if len(_cles_pos) else '-'} / événements {min(int(e['frame']) for e in _ev_ok)}-{max(int(e['frame']) for e in _ev_ok)}"
          f" | x max {max(float(e.get('x') or 0) for e in _ev_ok):.0f} | rattachés : "
          + ", ".join(f"décalage {d} éch. {e:.2f} → {len(v)}" for (d, e), v in _essais.items()))
TOUCHES.sort()
_fus = []
for t, ty in TOUCHES:                              # une touche = événements à moins de 2 s regroupés
    if _fus and t - _fus[-1][1] <= 2.0:
        _fus[-1][1] = t; _fus[-1][2].add(ty)
    else:
        _fus.append([t, t, {ty}])
print(f"   TOUCHES du {NUMERO} {COULEUR} : {len(TOUCHES)} événement(s) → {len(_fus)} touche(s) regroupée(s)")
for a, b, tys in _fus[:80]:
    print(f"     {fmt(a)}{'-' + fmt(b) if b - a >= 1 else ''} : {', '.join(sorted(tys))}")
with open(f"{SORTIE}/touches_v31_{MATCH}_{NUMERO}_{COULEUR.replace(' ', '_')}.csv", "w", encoding="utf-8") as _f:
    _f.write("debut;fin;types\n" + "".join(f"{fmt(a)};{fmt(b)};{'+'.join(sorted(t))}\n" for a, b, t in _fus))
# ── joueur seul : BUTS où le joueur est IMPLIQUÉ ; les autres (ex. un but du 9 rouge sans lui) sont ignorés ──
# Impliqué = une de SES touches dans les 15 s avant le but (but, passe, centre, duel…) OU reconnu dans les 8 s avant
# (dans l'action : buteur de la tête, défenseur sur le tireur…). Simplement visible 20 s avant ne suffit plus.
_buts_joueur = []
_FICHE_BUTS_CONF, _FICHE_BUTS_POSS = [], []     # 18/11 : pour la fiche « Bilan du joueur »
# 19/11 : secondes montrées AVANT l'heure d'un but. Un but non confirmé est souvent daté d'après le coup d'envoi qui
# suit (heure du but inconnue) : si le coup d'envoi a tardé (changement de joueur…), le but est bien plus tôt
# (SRU : but réel 89:52, daté 90:53 → la séquence ne montrait que le replacement des joueurs).
_AVANT_BUT = {}
# ── 19/11 (9) : HEURES DE BUT VUES par l'analyse du match (réponses Gemini déjà payées, gemini_cache_<match>*.jsonl) :
#    validation des coups d'envoi (GOAL_TIME), classification des temps forts (OUTCOME GOAL, ACTION_TIME), et coups de
#    pied arrêtés (RESTART_TYPE). Sert à montrer un but non confirmé à partir de sa VRAIE heure (pas 70 s à l'aveugle).
def _reponses_analyse():
    import glob as _g_ra, re as _re_ra
    buts_vus, arrets_jeu = [], []
    _bv = os.path.basename(VIDEO)
    for _f in _cherche(f"gemini_cache_{MATCH}*.jsonl") + \
              _g_ra.glob(f"{SORTIE}/gemini_cache_{MATCH}*.jsonl"):
        for _l in open(_f, encoding="utf-8"):
            try:
                _d = json.loads(_l)
            except ValueError:
                continue
            _c = _d.get("contexte") or {}
            if os.path.basename(str(_c.get("video_path", ""))) != _bv:
                continue
            _t = _d.get("text") or ""
            if _c.get("fonction") == "validate_kickoff_context_gemini":
                _k = float(_c.get("kickoff_time_abs") or 0)
                _gt_ = _re_ra.search(r"GOAL_TIME:\s*\**\s*[A-Z ]*?(\d+):(\d\d)(?::(\d\d))?", _t)
                if _gt_ and _re_ra.search(r"GOAL_DECISION:\s*\**\s*GOAL\b", _t):
                    h = (int(_gt_.group(1)) * 3600 + int(_gt_.group(2)) * 60 + int(_gt_.group(3))) if _gt_.group(3) \
                        else int(_gt_.group(1)) * 60 + int(_gt_.group(2))
                    if _k - 120 <= h <= _k + 10:          # heure plausible (pas « 07:01 » pour 73:01)
                        buts_vus.append(float(h))
                _rt = _re_ra.search(r"RESTART_TYPE:\s*\**\s*([A-Z_]+)", _t)
                if _rt and _rt.group(1) in ("GOAL_KICK", "FREE_KICK", "CORNER"):
                    arrets_jeu.append((_k, _rt.group(1), _t.lower()))
            elif _c.get("fonction") == "classify_highlight_window":
                _o = _re_ra.search(r"OUTCOME:\s*\**\s*GOAL\b", _t); _a = _re_ra.search(r"ACTION_TIME:\s*\**\s*(\d+):(\d\d)", _t)
                if _o and _a:
                    buts_vus.append(float(int(_a.group(1)) * 60 + int(_a.group(2))))
    return sorted(set(buts_vus)), arrets_jeu


_BUTS_VUS, _ARRETS_JEU = _reponses_analyse()


def _avant_but(g, confirme):
    """Secondes à montrer avant l'heure annoncée d'un but : 25 s s'il est confirmé ou si l'analyse voit un but à cette
    heure ; sinon jusqu'au but vu par l'analyse le plus proche dans les 90 s précédentes (+ 20 s), ou 35 s."""
    if confirme or any(g - 10 <= t <= g + 5 for t in _BUTS_VUS):     # heure confirmée par l'analyse elle-même
        return 25.0
    vus = [t for t in _BUTS_VUS if g - 90 <= t < g - 10]
    return (g - max(vus) + 20.0) if vus else 35.0                       # le but vu le PLUS PROCHE avant

# 19/11 (8) : couleur de l'ADVERSAIRE en anglais (comme dans les réponses de l'analyse du match)
_FR_EN = {"rouge": "red", "bleu": "blue", "bleu clair": "light blue", "blanc": "white", "noir": "black", "jaune": "yellow",
          "vert": "green", "orange": "orange", "violet": "purple", "gris": "grey"}
_ADV_EN = _FR_EN.get(COULEUR_EQUIPE.get(1 - EQ, ""), "") if GARDIEN else ""
if BOUT_EN_BOUT and BOUT_EN_BOUT.get("buts"):
    _t_pos = np.array(sorted(KO + n / FPS for n in positions)) if positions else np.array([])
    print(f"\n   BUTS du match ({len(BOUT_EN_BOUT['buts'])}) — 18/11 : CONFIRMÉS toujours dans le résumé ; POSSIBLES "
          f"(non confirmés) : TOUS dans le résumé du joueur (19/11), avec son lien au but :")
    for b in BOUT_EN_BOUT["buts"]:
        g = float(b["time_abs"])
        vu = float(((_t_pos >= g - 8) & (_t_pos <= g + 2)).sum()) / 6.0 if len(_t_pos) else 0.0   # ≈ secondes (6 images/s)
        touche = [t for t, _ in TOUCHES if g - 15 <= t <= g + 1]
        garde = bool(touche) or vu >= 1.0
        raison = (f"{len(touche)} touche(s) dans les 15 s avant" if touche else f"reconnu ≈ {vu:.0f} s dans les 8 s avant") if garde \
            else "ignoré (pas impliqué, ou pas reconnu)"
        # 08/11 (décision de l'utilisateur) : TOUS les buts du match entrent dans le résumé du joueur, pour ne jamais
        # rater un but où il est buteur ou impliqué ; l'indication « impliqué » n'est plus qu'une information.
        _conf_b = b.get("statut") == "confirmé"
        # 19/11 (11) : décision de l'utilisateur — TOUS les buts du match (confirmés ET possibles, des deux équipes) entrent
        # dans le résumé du joueur : même avec peu d'actions, il reçoit au moins un résumé du match. La raison indique
        # son lien avec le but (impliqué, son équipe, adversaire) ; les buts possibles restent marqués « non confirmé ».
        _eq_b = str(b.get("equipe") or "").lower()
        _SON_EN = _FR_EN.get(COULEUR_EQUIPE.get(EQ, ""), "")
        _ADV2 = _FR_EN.get(COULEUR_EQUIPE.get(1 - EQ, ""), "")
        if garde:
            lien = "impliqué : " + raison
        elif _SON_EN and _SON_EN in _eq_b:
            lien = "but de son équipe"
        elif _ADV2 and _ADV2 in _eq_b:
            lien = "but encaissé" if GARDIEN else "but de l'adversaire"
        else:
            lien = "équipe non identifiée"
        _buts_joueur.append(g)
        _AVANT_BUT[g] = _avant_but(g, _conf_b)       # 19/11 (9) : jusqu'au but vu par l'analyse, sinon 35 s
        (_FICHE_BUTS_CONF if _conf_b else _FICHE_BUTS_POSS).append((g, lien.replace("impliqué : ", "")))
        print(f"     but {fmt(g)} ({b.get('statut')}) → dans le résumé | {lien}")
# ── joueur seul : CE QUE L'ON SAIT DE SON MATCH (données disponibles) ──
_types = collections.Counter(ty for _, ty in TOUCHES)
_presence = len(positions) / 6.0 if positions else 0.0
print(f"\n   DONNÉES DU {NUMERO} {COULEUR} : présence reconnue ≈ {_presence / 60:.1f} min | {len(_fus)} action(s) avec ballon "
      f"({len(TOUCHES)} événements : " + ", ".join(f"{k} {v}" for k, v in _types.most_common()) + f") | {len(_buts_joueur)} but(s) du match dans le résumé")
print("   (fautes : NON détectées par le suivi — à ajouter plus tard si utile)")
# ── 19/11 (8) : GARDIEN — ses ARRÊTS et les TIRS ADVERSES, repérés par l'analyse du match (classification Gemini des
#    temps forts, déjà payée) : ils ne dépendent pas de sa visibilité. Lus dans gemini_cache_<match>*.jsonl en entrée.
_ARRETS, _TIRS_ADV = [], []
if GARDIEN:
    import glob as _g_cl, re as _re_cl
    _vu_t = set()
    _bvid = os.path.basename(VIDEO)
    for _f in _cherche(f"gemini_cache_{MATCH}*.jsonl") + \
              _g_cl.glob(f"{SORTIE}/gemini_cache_{MATCH}*.jsonl"):
        for _l in open(_f, encoding="utf-8"):
            try:
                _d = json.loads(_l)
            except ValueError:
                continue
            _cx_ = _d.get("contexte") or {}
            if _cx_.get("fonction") != "classify_highlight_window" or os.path.basename(str(_cx_.get("video_path", ""))) != _bvid:
                continue
            _t = _d.get("text") or ""
            _o = _re_cl.search(r"OUTCOME:\s*\**\s*([A-Z_]+)", _t); _a = _re_cl.search(r"ACTION_TIME:\s*\**\s*(\d+):(\d+)", _t)
            _c = _re_cl.search(r"TEAM_COLOR:\s*\**\s*([A-Za-z ]+)", _t)
            if not (_o and _a and _c):
                continue
            _oc, _ta, _col = _o.group(1), int(_a.group(1)) * 60 + int(_a.group(2)), _c.group(1).strip().lower()
            if _oc not in ("SAVE", "WIDE", "POST", "BLOCKED") or not _ADV_EN or _col != _ADV_EN:
                continue
            if any(abs(_ta - t) <= 10 for t in _vu_t):
                continue
            _vu_t.add(_ta)
            (_ARRETS if _oc == "SAVE" else _TIRS_ADV).append((float(_ta), _oc))
    _ARRETS.sort(); _TIRS_ADV.sort()
    print(f"\n   GARDIEN — arrêts repérés par l'analyse du match (tirs « {_ADV_EN} » arrêtés) : {len(_ARRETS)} "
          f"({', '.join(fmt(t) for t, _ in _ARRETS)}) | autres tirs adverses (à côté, poteau, contrés) : {len(_TIRS_ADV)}")
    TOUCHE_AVANT, TOUCHE_APRES = 8.0, 10.0      # gardien : voir l'action entière (coup franc jusqu'au tir)
    # 19/11 (9) : retirer ses RELANCES (6 mètres, dégagements) — demande de l'utilisateur. Une action du gardien est gardée
    # si elle est près d'un tir adverse ou d'un but (≤ 20 s), ou près d'un coup de pied arrêté ADVERSE dangereux (coup
    # franc ou corner près de sa surface, ≤ 30 s, d'après l'analyse du match) ; sinon c'est une relance → retirée.
    _proches = [t for t, _ in _ARRETS + _TIRS_ADV] + list(_buts_joueur)
    _danger = [k for k, rt, txt in _ARRETS_JEU if rt in ("FREE_KICK", "CORNER")
               and any(m in txt for m in ("penalty", "box", "wall", "area", "surface"))
               and not any(m in txt for m in ("defensive", "goal kick", "goalkeeper", "clearance"))]   # pas SES relances
    _six = [k for k, rt, _ in _ARRETS_JEU if rt == "GOAL_KICK"]
    _gardees, _retirees = [], []
    for a_, b_, ty_ in _fus:
        m_ = (a_ + b_) / 2
        if any(abs(m_ - t) <= 20 for t in _proches) or any(abs(m_ - k) <= 30 for k in _danger):
            _gardees.append((a_, b_, ty_))
        else:
            _retirees.append((a_, b_, "6 mètres" if any(abs(m_ - k) <= 40 for k in _six) else "relance / action isolée"))
    print(f"   GARDIEN — ses actions : {len(_gardees)} gardée(s) (près d'un tir, d'un but ou d'un coup de pied arrêté adverse) "
          f"| {len(_retirees)} retirée(s) : " + (", ".join(f"{fmt(a_)} ({r_})" for a_, _, r_ in _retirees) or "aucune"))
    _fus = _gardees
_fiche_joueur(True, presence_s=_presence, n_passages=len(retenus), n_moments=len(_fus),
              buts_conf=_FICHE_BUTS_CONF, buts_poss=_FICHE_BUTS_POSS,
              **({"lignes_supp": [("Arrêts repérés par l'analyse", str(len(_ARRETS))),
                                  ("Autres tirs adverses", f"{len(_TIRS_ADV)} (à côté, poteau ou contrés)")],
                  "source_presence": "passages où il est reconnu : maillot vérifié par Gemini",
                  } if GARDIEN else {}))
if MONTAGE == "touches" and (_fus or _buts_joueur or _ARRETS):
    _FEN = sorted([(a - TOUCHE_AVANT, b + TOUCHE_APRES) for a, b, _ in _fus] + [(g - _AVANT_BUT.get(g, 25.0), g + 5.0) for g in _buts_joueur]
                  + [(t - 12.0, t + 6.0) for t, _ in _ARRETS + _TIRS_ADV])   # 25 s avant : l heure du but du pipeline peut être en retard (jusqu à ~20 s)
    print(f"   montage : {len(_fus)} touche(s) + {len(_buts_joueur)} but(s) avec le joueur")
elif MONTAGE == "touches":
    print("   ⚠️ aucune touche ni aucun but rattaché au joueur : montage de présence (v2.8)")
    _FEN = []                                      # sinon le montage reprendrait les mi-temps entières

# ── 5. (suite) CONSENSUS DANS LE TEMPS (option, désactivée par défaut) ───────────────────────────
CONS_GROUPES, CONS_AJOUT = [], 0
if CONSENSUS_TEMPORAL:
    renf = {}
    for _c in sorted(_cherche(f"lectures_renforcees_{MATCH}.json")) + \
              [f"{SORTIE}/lectures_renforcees_{MATCH}.json"]:
        if os.path.exists(_c):
            try:
                renf.update(json.load(open(_c, encoding="utf-8")))
            except Exception as _e:
                print(f"   ⚠️ {_c} illisible ({_e})")
    autre_eq = 1 - EQ
    obs = []
    for src_l, base in (("normale", lectures), ("renforcée", renf)):
        for sid, e in base.items():
            if sid not in segments or e.get("numeros") is None or e.get("non_lu"):
                continue
            for n, num, p in zip(e.get("images", []), e["numeros"], e.get("parts", [])):
                if n in segments[sid]["pos"]:
                    obs.append((int(n), sid, num, equipe_mesuree(p), tuple(segments[sid]["pos"][n][0]), src_l))
    obs = sorted(set(obs), key=lambda o: (o[0], o[1], o[5], -1 if o[2] is None else o[2], -1 if o[3] is None else o[3]))
    dix = [o for o in obs if o[2] == NUMERO and o[3] != autre_eq]

    def _dist_h(b1, b2):
        h = max(1.0, ((b1[3] - b1[1]) + (b2[3] - b2[1])) / 2)
        return float(np.hypot((b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2, b1[3] - b2[3]) / h)

    def _lies(o1, o2):
        dt = abs(o1[0] - o2[0]) / FPS
        return dt <= CONS_S and _dist_h(o1[4], o2[4]) <= CONS_BASE_H + CONS_H_PAR_S * dt

    parent = list(range(len(dix)))
    def _rac(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(dix)):
        for j in range(i + 1, len(dix)):
            if (dix[j][0] - dix[i][0]) / FPS > CONS_S:
                break
            if _lies(dix[i], dix[j]):
                parent[_rac(j)] = _rac(i)
    groupes = collections.defaultdict(list)
    for i in range(len(dix)):
        groupes[_rac(i)].append(dix[i])
    coupures = collections.defaultdict(list)
    for o in obs:
        if o[3] == autre_eq or (o[2] is not None and o[2] != NUMERO):
            coupures[o[1]].append(o[0])
    deja = set(positions)
    ajout = {}
    for k, g in enumerate(sorted(groupes.values(), key=lambda g: g[0][0]), 1):
        # lectures simultanées à deux endroits : exclure la moins soutenue (les deux si égalité)
        exclues = set()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if abs(g[i][0] - g[j][0]) <= 0.2 * FPS and _dist_h(g[i][4], g[j][4]) > 0.5:
                    si = sum(1 for o in g if o is not g[i] and _lies(o, g[i]))
                    sj = sum(1 for o in g if o is not g[j] and _lies(o, g[j]))
                    if si <= sj: exclues.add(i)
                    if sj <= si: exclues.add(j)
        garde = [o for i, o in enumerate(g) if i not in exclues]
        n_rouges = sum(1 for o in garde if o[3] == EQ)
        valide = len(garde) >= CONS_MIN_LECTURES and n_rouges >= CONS_MIN_ROUGES
        nouv = set()
        if valide:
            for n, sid, *_ in garde:
                lo, hi = n - GARDE_S * FPS, n + GARDE_S * FPS
                for c in coupures.get(sid, []):
                    if c < n: lo = max(lo, c + 1)
                    elif c > n: hi = min(hi, c - 1)
                for m in segments[sid]["pos"]:
                    if lo <= m <= hi and m not in deja and m not in ajout:
                        ajout[m] = segments[sid]["pos"][m][0]; nouv.add(m)
        statut = ("ajouté" if nouv else "déjà couvert") if valide else "faible (non ajouté)"
        if len(g) >= 2 or exclues:
            CONS_GROUPES.append({"groupe": k, "debut": fmt(KO + g[0][0] / FPS), "fin": fmt(KO + g[-1][0] / FPS),
                                 "lectures": len(garde), "rouges": n_rouges, "fragments": len({o[1] for o in garde}),
                                 "exclues_conflit": len(exclues), "statut": statut,
                                 "secondes_ajoutees": len({int(KO + m / FPS) for m in nouv}),
                                 "detail": " ".join(f"{fmt(KO + o[0] / FPS)}:{o[1]}:{'R' if o[3] == EQ else '?'}:{o[5][:4]}"
                                                    for o in garde),
                                 "_obs": garde})
    for m, bb in ajout.items():
        positions[m] = (bb, "consensus")
    blocs, cur = [], []
    for m in sorted(ajout):
        if cur and m - cur[-1] > 0.5 * FPS:
            blocs.append(cur); cur = []
        cur.append(m)
    if cur:
        blocs.append(cur)
    for bl in blocs:
        spans.append((KO + bl[0] / FPS, KO + bl[-1] / FPS))
    CONS_AJOUT = len({int(KO + m / FPS) for m in ajout})
    st = collections.Counter(gr["statut"] for gr in CONS_GROUPES)
    print(f"\n   CONSENSUS DANS LE TEMPS : {len(obs)} vignettes lues ({sum(1 for o in obs if o[5] == 'renforcée')} renforcées) | "
          f"groupes {dict(st)} | lectures exclues pour conflit : {sum(gr['exclues_conflit'] for gr in CONS_GROUPES)} | "
          f"présence ajoutée : {CONS_AJOUT} s")
    with open(f"{SORTIE}/consensus_v28_{MATCH}_{NUMERO}_{COULEUR}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[c for c in CONS_GROUPES[0] if c != "_obs"] if CONS_GROUPES else ["groupe"],
                           delimiter=";", extrasaction="ignore")
        w.writeheader(); w.writerows(CONS_GROUPES)
    lignes_c = []
    for gr in CONS_GROUPES:
        if gr["statut"] != "ajouté":
            continue
        cases = []
        for o in gr["_obs"][:7]:
            ok, img = image_video(o[0])
            if ok and decoupe(img, o[4]) is not None:
                cases.append(case(decoupe(img, o[4]), f"{fmt(KO + o[0] / FPS)} {'R' if o[3] == EQ else '?'}", o[3] == EQ))
        if cases:
            lignes_c.append(ligne_planche(f"groupe {gr['groupe']} | {gr['debut']}-{gr['fin']} | {gr['lectures']} lectures, "
                                          f"{gr['rouges']} rouges, {gr['exclues_conflit']} exclue(s) | +{gr['secondes_ajoutees']} s",
                                          (0, 140, 255), cases))
    if lignes_c:
        ecrire_planches(lignes_c, f"{SORTIE}/verification_consensus_v30")
        print(f"   planche : verification_consensus_v30_p*.jpg ({len(lignes_c)} groupe(s) ajouté(s), à vérifier)")
cles_pos = np.array(sorted(positions))
sequences = []
if _FEN:                                  # v2.9 : le résumé complet, repère là où le joueur est reconnu
    spans = list(_FEN)
for a, b in sorted(spans):
    if sequences and a - sequences[-1][1] <= FUSION_S:
        sequences[-1][1] = max(sequences[-1][1], b)
    else:
        sequences.append([a, b])


def cadre_a(t_video):
    num = (t_video - KO) * FPS
    i = int(np.argmin(np.abs(cles_pos - num)))
    if abs(cles_pos[i] - num) > 0.3 * FPS:
        return None
    return positions[int(cles_pos[i])]


def extrait_avec_repere(a, b, sortie):
    cap_e = cv2.VideoCapture(VIDEO)
    fps_v = cap_e.get(cv2.CAP_PROP_FPS) or FPS
    cap_e.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    ok, img = cap_e.read()
    if not ok:
        cap_e.release(); return False, 0.0, 0.0
    H, W = img.shape[:2]
    tmp = sortie + ".brut.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps_v, (W, H))
    t, dernier, t_dernier, n_lu, n_rac, n_cons = a, None, None, 0, 0, 0
    while ok and t <= b:
        x = cadre_a(t)
        if x is not None:
            dernier, t_dernier = x, t                   # seule une VRAIE position remet le compteur à zéro
        elif dernier is not None and t - t_dernier <= TENUE_REPERE_S:
            x = dernier                                 # petit trou : repère tenu au plus TENUE_REPERE_S
        if x is not None:
            bb, source = x
            est_rac = source.startswith("raccord")
            est_cons = source == "consensus"
            rac = est_rac and MARQUER_RACCORDS
            n_rac += est_rac; n_cons += est_cons; n_lu += not (est_rac or est_cons)
            coul = (255, 160, 0) if rac else (0, 140, 255) if est_cons else (0, 230, 255)   # bleu / orange / jaune
            x1, y1, x2, y2 = [int(v) for v in bb]
            cx, haut = (x1 + x2) // 2, max(30, y1 - 12)
            cv2.drawContours(img, [np.array([[cx - 14, haut - 24], [cx + 14, haut - 24], [cx, haut]])], 0, coul, -1)
            txt = f"#{NUMERO}" + (f"? {source.split(':', 1)[1]}".rstrip() if rac else "*" if est_cons else "")
            cv2.putText(img, txt, (cx - 22, haut - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, coul, 2)
            cv2.ellipse(img, (cx, y2), (max(14, (x2 - x1) // 2 + 6), 8), 0, 0, 360, coul, 2)
        vw.write(img)
        ok, img = cap_e.read()
        t += 1.0 / fps_v
    vw.release(); cap_e.release()
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-ss", f"{a:.2f}", "-t", f"{b - a:.2f}", "-i", VIDEO,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                        "-c:a", "aac", "-shortest", "-movflags", "+faststart", sortie], capture_output=True)
    os.remove(tmp)
    return r.returncode == 0 and os.path.exists(sortie), n_lu / fps_v, n_rac / fps_v, n_cons / fps_v


# ── 5 bis. CONTRÔLE DES BUTS (vérité terrain) : le joueur est-il repéré juste avant chaque but ? ─────────
CONTROLE_BUTS = []
_vt = sorted(_cherche("verite_terrain.json")) + \
      sorted(glob.glob(f"{SORTIE}/verite_terrain.json"))
_buts, _buteurs = [], {}
for _c in _vt:
    try:
        _m = json.load(open(_c, encoding="utf-8"))["matchs"][MATCH]
        _buts, _buteurs = _m.get("buts", []), _m.get("buteurs", {})
        break
    except Exception:
        pass
if not _buts:
    print("\n   contrôle des buts : verite_terrain.json absent ou sans ce match (ajoute-le en entrée) — ignoré")
else:
    _sec = lambda t: int(t.split(":")[0]) * 60 + float(t.split(":")[1])
    print(f"\n   CONTRÔLE DES BUTS ({len(_buts)}) : présence du {NUMERO} {COULEUR} dans les {CONTROLE_BUT_AVANT_S} s avant chaque but")
    for _b in _buts:
        tb = _sec(_b)
        n0, n1 = (tb - CONTROLE_BUT_AVANT_S - KO) * FPS, (tb + 1 - KO) * FPS
        dans = [n for n in positions if n0 <= n <= n1]
        s_lu = sum(1 for n in dans if positions[n][1] == "lu") * 5 / FPS
        s_rac = sum(1 for n in dans if positions[n][1].startswith("raccord")) * 5 / FPS
        s_cons = sum(1 for n in dans if positions[n][1] == "consensus") * 5 / FPS
        dernier = fmt(KO + max(dans) / FPS) if dans else "-"
        # segments de l'équipe dans la fenêtre, avec leur état
        etat = collections.Counter()
        en_attente = []
        for _, sid, _ in candidats:
            a_, b_ = min(segments[sid]["pos"]), max(segments[sid]["pos"])
            if b_ < n0 or a_ > n1:
                continue
            e = lectures.get(sid) or {}
            st = "retenu" if sid in retenus else e.get("non_lu") or ("lu" if e.get("numeros") is not None else "non examiné")
            etat[st] += 1
            if st in ("plafond", "non examiné"):
                en_attente.append(f"{fmt(KO + a_ / FPS)}-{fmt(KO + b_ / FPS)}")
        info = _buteurs.get(_b, {})
        qui = f"{info.get('equipe', '?')} {info.get('numero', '?')}" if info else "?"
        CONTROLE_BUTS.append([_b, qui, info.get("remarque", ""), round(s_lu, 1), round(s_rac, 1), dernier,
                              dict(etat), " ".join(en_attente)])
        print(f"     but {_b} (buteur : {qui}) : {NUMERO} repéré {s_lu:.0f} s (lu) + {s_rac:.0f} s (raccord)"
              + (f" + {s_cons:.0f} s (consensus)" if CONSENSUS_TEMPORAL else "") + ", "
              f"dernière position {dernier} | segments rouges de la fenêtre : {dict(etat)}"
              + (f" | encore à lire : {', '.join(en_attente)}" if en_attente else ""))
    with open(f"{SORTIE}/controle_buts_v30_{MATCH}_{NUMERO}_{COULEUR}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["but", "buteur", "remarque", "repere_lu_s", "repere_raccord_s", "derniere_position",
                    "segments_rouges_fenetre", "segments_a_lire"])
        w.writerows(CONTROLE_BUTS)

dossier_clips = f"{SORTIE}/clips_v28_{NUMERO}_{COULEUR}"
os.makedirs(dossier_clips, exist_ok=True)
for ancien in glob.glob(f"{dossier_clips}/seq_*.mp4"):
    os.remove(ancien)
clips, lignes = [], []
for i, (a0, b0) in enumerate(sequences, 1):
    a, b = max(0.0, a0 - MARGE_AVANT), b0 + MARGE_APRES
    clip = f"{dossier_clips}/seq_{i:02d}.mp4"
    ok_c, s_lu, s_rac, s_cons = extrait_avec_repere(a, b, clip)
    if ok_c:
        clips.append(clip)
    lignes.append([i, fmt(a), fmt(b), f"{max(0, a0 - KO) / 60:.0f}'", round(b - a, 1), round(s_lu, 1), round(s_rac, 1),
                   round(s_cons, 1)])
    print(f"     séquence {i:2d} : vidéo {fmt(a)} → {fmt(b)} ({b - a:.0f} s, repère lu {s_lu:.0f} s, raccord {s_rac:.0f} s"
          + (f", consensus {s_cons:.0f} s" if CONSENSUS_TEMPORAL else "") + ")")
for c in _caps:
    c.release()
suffixe = "_essai" if MODE == "essai" else ""
montage = f"{SORTIE}/montage_v30_{MATCH}_{NUMERO}_{COULEUR}{suffixe}.mp4"
if clips:
    with open(f"{dossier_clips}/liste.txt", "w") as f:
        f.writelines(f"file '{c}'\n" for c in clips)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{dossier_clips}/liste.txt", "-c", "copy", montage], capture_output=True)
with open(f"{SORTIE}/montage_v30_{MATCH}_{NUMERO}_{COULEUR}{suffixe}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["sequence", "debut_video", "fin_video", "minute_de_jeu", "duree_s", "repere_lu_s", "repere_raccord_s",
                "repere_consensus_s"])
    w.writerows(lignes)

# ── BILAN ─────────────────────────────────────────────────────────────────────────────────────
duree = sum(l[4] for l in lignes)
minutes = sorted({l[3] for l in lignes}, key=lambda m: int(m[:-1]))
print(f"\n{'═' * 90}\nBILAN v2.8 du {NUMERO} {COULEUR} ({MATCH}, mode {MODE})")
print(f"   lectures fusionnées : {len(SOURCES_L)} fichier(s) | conflits journalisés : {len(CONFLITS)}")
print(f"   segments candidats : {N_CANDIDATS_TOTAL} | examinés : {len(candidats)} | lectures réutilisées : {N_REUTILISEES_ICI} | "
      f"questions évitées déjà connues : {N_EVITEES_CONNUES}")
print(f"   segments examinés par le lecteur de numéros : {compteur['questions']} | évités par la couleur : {compteur['evitees']} | "
      f"en attente (plafond) : {compteur['plafond']} | erreurs : {len(erreurs)}")
print(f"   segments retenus par lecture : {len(retenus)} | raccords ({RACCORD_MONTAGE}, montage) : {len(raccords)} | "
      + " | ".join(f"{c[0]} : {c[1]}" for c in COMPARAISON))
print(f"   montage : {len(clips)} séquence(s), {duree / 60:.1f} min | repère lu {sum(l[5] for l in lignes):.0f} s, "
      f"raccord {sum(l[6] for l in lignes):.0f} s" + (f", consensus {sum(l[7] for l in lignes):.0f} s" if CONSENSUS_TEMPORAL else "")
      + f" | minutes de jeu : {', '.join(minutes)}")
print(f"   consensus dans le temps : {'ACTIVÉ' if CONSENSUS_TEMPORAL else 'désactivé (comportement v2.7)'}"
      + (f" | groupes ajoutés : {sum(1 for g in CONS_GROUPES if g['statut'] == 'ajouté')}, faibles écartés : "
         f"{sum(1 for g in CONS_GROUPES if g['statut'].startswith('faible'))}, présence ajoutée {CONS_AJOUT} s"
         if CONSENSUS_TEMPORAL else ""))
print(f"   erreurs de raccordement : NON mesurables automatiquement → vérifier les repères BLEUS du montage et "
      f"verification_raccords_v28_p*.jpg\n   {montage if os.path.exists(montage) else '(aucun montage)'}\n"
      f"   Durée du notebook : {(time.time() - T0) / 60:.1f} min")
cout_d = dollars(COUT)
print(f"   réflexion de la lecture des numéros : {REFLEXION_LECTURE or 'normale'}")
print(f"   COÛT GEMINI de ce run (mesuré) : {COUT['questions']} question(s) | jetons entrée {COUT['entree']:,} | sortie "
      f"{COUT['sortie']:,} | réflexion {COUT['reflexion']:,} | ≈ {cout_d:.3f} $ (≈ {cout_d * TAUX_EUR:.3f} €)"
      + (f" | ⚠️ {COUT['sans_mesure']} réponse(s) sans mesure" if COUT["sans_mesure"] else ""))
for usage, c in COUT_PAR_USAGE.items():
    print(f"      {usage} : {c['questions']} question(s), ≈ {dollars(c):.3f} $"
          + (f" ({dollars(c) / c['questions'] * 1000:.2f} $ pour 1 000 questions)" if c["questions"] else ""))
print("═" * 90)
# journal des coûts : une ligne par run, conservé d'un run à l'autre s'il est remis en entrée
journal = f"{SORTIE}/cout_gemini_{MATCH}.csv"
anciens = sorted(_cherche(f"cout_gemini_{MATCH}.csv"))
if not os.path.exists(journal) and anciens:
    import shutil
    shutil.copy(anciens[-1], journal)
nouveau = not os.path.exists(journal)
with open(journal, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    if nouveau:
        w.writerow(["date", "version", "mode", "questions", "jetons_entree", "jetons_sortie", "jetons_reflexion",
                    "cout_dollars", "cout_euros_approx"])
    w.writerow([time.strftime("%Y-%m-%d %H:%M"), VERSION, MODE, COUT["questions"], COUT["entree"], COUT["sortie"],
                COUT["reflexion"], round(cout_d, 4), round(cout_d * TAUX_EUR, 4)])
print(f"   journal des coûts : {journal} (une ligne par run)")


# ── BOUT EN BOUT : bilan global ──
if BOUT_EN_BOUT:
    print(f"\n{'═' * 90}\nBILAN DE BOUT EN BOUT — {MATCH}, joueur n°{NUMERO} {COULEUR}")
    print(f"   durée : analyse du match {BOUT_EN_BOUT.get('duree_analyse_min')} min + montage du joueur "
          f"{(_tbe.time() - T_DEBUT_MONTAGE) / 60:.0f} min")
    print("   coût Gemini : analyse = « [GEMINI CACHE] … payée(s) » + « coût tenues + attribution » (cellule 1) ;"
          " joueur = « COÛT GEMINI de ce run » + « CONFIRMATION GEMINI » (ci-dessus)")
    print(f"   livrables : rapport PDF (cellule 1), montage {SORTIE}/montage_v30_{MATCH}_{NUMERO}_{COULEUR}.mp4\n{'═' * 90}")


# ════════════════════════════════════════════════════════════════════════════════════════════
# BILAN DE PRODUCTION — un match, de la vidéo brute aux livrables
# ════════════════════════════════════════════════════════════════════════════════════════════
_duree_montage = (time.time() - T_DEBUT_MONTAGE) / 60
_cout_an = sum(v[1] for v in COUT_ANALYSE.values())
_cout_j = (cout_d if "cout_d" in dir() else 0.0) + (_jet["e"] / 1e6 * 0.75 + _jet["s"] / 1e6 * 3.75 if "_jet" in dir() else 0.0) \
          + (_cout3 if "_cout3" in dir() else 0.0)        # 19/11 : + vérification Gemini du gardien (oubliée au 1er run)
# 11/11 : le coût joueur oubliait la relecture Gemini (_jet) — corrigé
print(f"\n{'█' * 90}\nBILAN DE PRODUCTION — {MATCH} (vidéo {os.path.getsize(VIDEO) / 1e9:.1f} Go) — DÉTECTEUR RF-DETR "
      f"{DETECTEUR_TAILLE} (Apache 2.0)\n{'█' * 90}")
print(f"   TEMPS   analyse du match {DUREE_ANALYSE_MIN:6.1f} min | montage du joueur n°{NUMERO} {_duree_montage:6.1f} min | "
      f"TOTAL {DUREE_ANALYSE_MIN + _duree_montage:6.1f} min")
if COUT_ANALYSE:
    print(f"   GEMINI  analyse {_cout_an:.2f} $ (dont équipes {sum(v[1] for f, v in COUT_ANALYSE.items() if f == '__call__'):.2f} $) | "
          f"joueur {_cout_j:.2f} $ | TOTAL {_cout_an + _cout_j:.2f} $ (≈ {(_cout_an + _cout_j) * 0.90:.2f} € au taux de la facture Google du 12/11 : 0,90 € pour 1 $)")
else:     # 18/11 : étape 2 lancée seule — le coût de l'analyse est dans le journal du run d'analyse
    print(f"   GEMINI  analyse : non mesurée dans ce run (voir le run d'analyse du match) | joueur {_cout_j:.2f} $")
try:
    import torch as _t_bilan
    _mat_bilan = "GPU T4 ×2" if _t_bilan.cuda.is_available() else "processeur"
except Exception:
    _mat_bilan = "matériel inconnu"
print(f"   MATÉRIEL {_mat_bilan} | montage du joueur {_duree_montage / 60:.2f} h"
      + (f" | analyse du match {DUREE_ANALYSE_MIN / 60:.2f} h (run d'analyse)" if DUREE_ANALYSE_MIN else ""))
print(f"   ÉQUIPES {_att.get('tenues') if _att else '—'} | score confirmé {_att.get('score_confirme') if _att else '—'} | "
      f"estimé {_att.get('score_estime') if _att else '—'}")
_livr = [f for f in (f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf", f"{SORTIE}/fiche_joueur_{MATCH}_{NUMERO}.pdf",
                     f"{SORTIE}/rapport_{MATCH}.pdf", f"{SORTIE}/montage_v30_{MATCH}_{NUMERO}_{COULEUR}.mp4",
                     f"{SORTIE}/cache_{MATCH}_reference.zip") if os.path.exists(f)]
print(f"   LIVRABLES (réellement produits) : {' | '.join(_livr) or 'aucun'}")   # 18/11 : plus de chemin inventé
print(f"   (comparer : tarifs Pro 2,69 € / Unique 5 €)\n{'█' * 90}")

# ── PRODUCTION : résultat pour monter_joueur() ──
_mont_p = f"{SORTIE}/montage_v30_{MATCH}_{NUMERO}_{COULEUR}.mp4"
_ecrire_resultat(trouve=True, montage=_mont_p if os.path.exists(_mont_p) else None,
                 fiche=f"{SORTIE}/fiche_joueur_{MATCH}_{NUMERO}.pdf" if os.path.exists(f"{SORTIE}/fiche_joueur_{MATCH}_{NUMERO}.pdf") else None,
                 rapport_et_fiche=f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf" if os.path.exists(f"{SORTIE}/rapport_{MATCH}_joueur{NUMERO}.pdf") else None,
                 gardien=bool(GARDIEN), presence_s=round(float(_presence), 1), passages=len(retenus), moments_ballon=len(_fus),
                 buts_confirmes=[round(g, 1) for g, _ in _FICHE_BUTS_CONF], buts_possibles=[round(g, 1) for g, _ in _FICHE_BUTS_POSS],
                 arrets=[t for t, _ in _ARRETS], tirs_adverses=[t for t, _ in _TIRS_ADV],
                 cout_gemini_usd=round(float(_cout_j), 4), duree_min=round(float(_duree_montage), 1), message=None)
print(f"   RÉSULTAT : {SORTIE}/resultat_montage_joueur.json")
