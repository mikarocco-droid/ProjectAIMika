"""
team_color_matching.py
===========================================================
V5.2 - Appariement (labellisation) des 2 equipes detectees par le
tracker (clustering KMeans non supervise, cv2.kmeans sur un espace
19D histogramme-teinte + LAB, voir analysis/player_reid.py
_calibrate_teams/_extract_color) avec les couleurs nommees confirmees
par l'utilisateur via la pre-analyse Gemini vision (pre_analyse_upload.py).

PRINCIPE IMPORTANT (decide explicitement suite a une question de
l'utilisateur) : ce module ne modifie JAMAIS le clustering du tracker
lui-meme (pas d'injection de "rouge"/"jaune" comme centres initiaux du
KMeans) - le tracker continue de separer les 2 equipes entierement a
partir de ses propres observations de pixels, sans biais externe.

Ce module intervient SEULEMENT APRES coup : il prend les 2 centroides
deja calibres par le tracker, et essaie de leur ASSIGNER le nom de
couleur le plus proche parmi ceux fournis par Gemini - sans jamais
pretendre que la teinte mesuree (ex. "bordeaux") est litteralement
identique a la reference (ex. "rouge"). Si la correspondance est trop
incertaine, AUCUN nom n'est force - un libelle generique est utilise a
la place, avec la couleur reellement mesuree affichee en clair.

CALIBRATION DU SEUIL : le seuil DISTANCE_MAX_CONFIANCE ci-dessous n'a
PAS ete valide empiriquement (aucun appel Gemini utilise pour ce
travail, a la demande explicite de l'utilisateur ce soir) - valeur de
depart raisonnable, a ajuster avec de vrais tests des que possible.
"""

import numpy as np
import cv2


# Valeur de depart, NON calibree empiriquement - cf. note ci-dessus.
DISTANCE_MAX_CONFIANCE = 30.0  # dans l'espace LAB (echelle ~0-100 par canal)

# Meme mapping que detect_teams_preview.py::_color_name_to_hex - garde
# ici une copie locale pour ne pas creer de dependance circulaire entre
# les 2 modules (a factoriser plus tard si besoin).
NOM_COULEUR_VERS_HEX = {
    "rouge":        "#cc2200",
    "rouge foncé":  "#880000",
    "bordeaux":     "#6b0f2a",
    "vert":         "#2d7a2d",
    "vert foncé":   "#1a4d1a",
    "bleu":         "#1a4dcc",
    "bleu marine":  "#0a1a4d",
    "bleu foncé":   "#0a1a4d",
    "noir":         "#1a1a1a",
    "blanc":        "#f0f0f0",
    "jaune":        "#e6cc00",
    "orange":       "#e67300",
    "violet":       "#6600cc",
    "rose":         "#cc0066",
    "gris":         "#808080",
}


def _hex_vers_lab(hex_color):
    """Convertit une couleur hex (#rrggbb) en LAB (via BGR, comme OpenCV)."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    bgr = np.uint8([[[b, g, r]]])
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return lab


def _nom_couleur_vers_lab(nom):
    """Convertit un nom de couleur (ex. Gemini: 'rouge', 'jaune') en LAB,
    via le mapping hex existant. Retourne None si nom inconnu."""
    if not nom:
        return None
    nom_lower = nom.lower()
    for cle, hex_val in NOM_COULEUR_VERS_HEX.items():
        if cle in nom_lower:
            return _hex_vers_lab(hex_val)
    return None


def _centroide_vers_lab(centroide_19d):
    """Extrait la couleur LAB approximative d'un centroide 19D du
    tracker (16 bins histogramme teinte + 3D LAB pondere x4).

    cf. analysis/player_reid.py::_extract_color pour le format exact :
    lab_feat = [(L/255), (A-128)/128, (B-128)/128] * 4.0
    On inverse cette transformation pour retrouver L, A, B ~ [0-255].
    """
    lab_scaled = centroide_19d[16:19]  # les 3 dernieres dimensions
    l = (lab_scaled[0] / 4.0) * 255.0
    a = (lab_scaled[1] / 4.0) * 128.0 + 128.0
    b = (lab_scaled[2] / 4.0) * 128.0 + 128.0
    return np.array([l, a, b], dtype=np.float32)


def apparier_couleurs_equipes(centroides_tracker, couleurs_gemini, confiances_gemini=None):
    """
    Tente d'apparier les 2 centroides du tracker (clustering deja fait,
    NON modifie par cette fonction) aux 2 noms de couleur confirmes par
    Gemini pendant la pre-analyse.

    Parametres
    ----------
    centroides_tracker : liste de 2 vecteurs 19D
        Les _team_centroids calibres par player_reid.py (cv2.kmeans),
        dans l'ordre [cluster0, cluster1].
    couleurs_gemini : liste de 2 str
        Les noms de couleur confirmes par l'utilisateur/Gemini, dans
        l'ordre [equipe_A, equipe_B] (ex. ["rouge", "jaune"]).
    confiances_gemini : liste de 2 str ou None
        Confiance Gemini pour chaque couleur ("high"/"medium"/"low"),
        si disponible (cf. detect_teams_preview.py::_ask_gemini_colors).
        Une confiance "low" desactive l'appariement pour cette couleur
        (jamais utilisee comme reference fiable).

    Retourne
    --------
    dict avec, pour chaque cluster (0 et 1) :
      - "nom_assigne" : str ou None (nom Gemini assigne, ou None si
        aucune correspondance suffisamment confiante)
      - "distance" : float (distance LAB au nom assigne, ou au plus
        proche nom teste si aucun n'est retenu)
      - "couleur_mesuree_lab" : le LAB reellement mesure par le tracker
        (toujours disponible, meme si aucun nom n'a pu etre assigne)
    """
    if confiances_gemini is None:
        confiances_gemini = [None, None]

    # Filtrer les couleurs Gemini de confiance insuffisante ("low") -
    # elles ne sont jamais utilisees comme reference fiable pour
    # l'appariement.
    couleurs_utilisables = [
        (nom, conf) for nom, conf in zip(couleurs_gemini, confiances_gemini)
        if conf != "low"
    ]

    lab_centroides = [_centroide_vers_lab(c) for c in centroides_tracker]

    resultat = {
        0: {"nom_assigne": None, "distance": None, "couleur_mesuree_lab": lab_centroides[0]},
        1: {"nom_assigne": None, "distance": None, "couleur_mesuree_lab": lab_centroides[1]},
    }

    if len(couleurs_utilisables) < 2:
        # Pas assez de couleurs Gemini fiables pour tenter un
        # appariement a 2 - on retourne les couleurs mesurees seules,
        # sans nom assigne.
        return resultat

    noms_gemini = [c[0] for c in couleurs_utilisables[:2]]
    lab_gemini = [_nom_couleur_vers_lab(n) for n in noms_gemini]

    if any(l is None for l in lab_gemini):
        # Un des noms Gemini n'est pas dans notre mapping connu -
        # appariement impossible, retour sans nom assigne.
        return resultat

    # Essayer les 2 appariements possibles, garder celui de distance
    # totale la plus faible (cf. discussion : plus robuste qu'un
    # test isole par cluster, car il n'y a que 2 equipes possibles).
    appariement_A = {
        0: np.linalg.norm(lab_centroides[0] - lab_gemini[0]),
        1: np.linalg.norm(lab_centroides[1] - lab_gemini[1]),
    }
    appariement_B = {
        0: np.linalg.norm(lab_centroides[0] - lab_gemini[1]),
        1: np.linalg.norm(lab_centroides[1] - lab_gemini[0]),
    }

    total_A = appariement_A[0] + appariement_A[1]
    total_B = appariement_B[0] + appariement_B[1]

    if total_A <= total_B:
        meilleur = {0: (noms_gemini[0], appariement_A[0]), 1: (noms_gemini[1], appariement_A[1])}
    else:
        meilleur = {0: (noms_gemini[1], appariement_B[0]), 1: (noms_gemini[0], appariement_B[1])}

    for cluster_id, (nom, distance) in meilleur.items():
        resultat[cluster_id]["distance"] = float(distance)
        if distance <= DISTANCE_MAX_CONFIANCE:
            resultat[cluster_id]["nom_assigne"] = nom
        # Sinon : nom_assigne reste None - correspondance jugee trop
        # incertaine, on ne force rien (cf. discussion "bordeaux" vs
        # "rouge" - mieux vaut un libelle generique qu'une etiquette
        # potentiellement fausse).

    return resultat
