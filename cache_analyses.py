# analysis/cache_analyses.py
# 17/11/2026 — RÉUTILISER L'ANALYSE D'UNE VIDÉO DÉJÀ TRAITÉE (idée de l'utilisateur) — MODULE SEUL, PAS ENCORE BRANCHÉ.
#
# Si deux joueurs d'un même match envoient la même vidéo, la 2e analyse peut repartir du suivi et des réponses Gemini
# de la 1re (le replay le fait déjà : quelques centimes au lieu d'environ 2,50 $ et 1 h 45 de GPU). Seule la partie
# propre au joueur (identification, montage) est refaite.
#
# Principes :
#   • la vidéo est reconnue par son CONTENU (empreinte), pas par son nom : taille du fichier + SHA-256 de 3 morceaux
#     de 4 Mo (début, milieu, fin). Rapide (12 Mo lus) et insensible au renommage. Une vidéo RÉENCODÉE (autre fichier)
#     n'est pas reconnue : elle est simplement analysée normalement.
#   • l'entrée est liée à la VERSION de l'analyse (VERSION_ANALYSE + profil de réglages) : après une amélioration du
#     pipeline, une ancienne analyse n'est plus resservie ;
#   • DURÉE DE CONSERVATION limitée (SCOUTIA_CACHE_ANALYSES_JOURS, 30 jours par défaut) : les entrées expirées sont
#     supprimées AVEC leurs fichiers (images de mineurs : RGPD — durée à fixer avec l'avis juridique et à écrire dans
#     les conditions d'utilisation) ;
#   • rien n'est fait en secret côté client : la vidéo est livrée dans le délai annoncé, sans simuler un traitement.
#
# Branchement prévu (APRÈS la validation des réglages, une modification à la fois) : dans la tâche d'analyse, avant
# run_pipeline → chercher(video) ; si trouvé → _match_data = replay.load_cache(entree["fichiers"]["suivi"]) et
# SCOUTIA_GEMINI_CACHE_IN = entree["fichiers"]["gemini"] ; après une analyse complète → enregistrer(...).
import hashlib
import json
import os
import shutil
import time

VERSION_ANALYSE = "2026-11-17"      # à changer à chaque évolution qui modifie les résultats
TAILLE_MORCEAU = 4 * 1024 * 1024


def dossier_cache():
    return os.environ.get("SCOUTIA_CACHE_ANALYSES_DIR", "").strip() or os.path.join("outputs", "cache_analyses")


def duree_conservation_s():
    try:
        return float(os.environ.get("SCOUTIA_CACHE_ANALYSES_JOURS", "") or 30) * 86400
    except ValueError:
        return 30 * 86400


def version_courante():
    """Version des résultats : version du code + profil de réglages actif (un autre profil = d'autres résultats)."""
    return f"{VERSION_ANALYSE}|profil={os.environ.get('SCOUTIA_PROFIL_KO', '').strip() or 'aucun'}"


def empreinte_video(chemin):
    """Empreinte du contenu : taille + SHA-256 du début, du milieu et de la fin du fichier."""
    taille = os.path.getsize(chemin)
    h = hashlib.sha256()
    h.update(str(taille).encode())
    with open(chemin, "rb") as f:
        for debut in sorted({0, max(0, taille // 2 - TAILLE_MORCEAU // 2), max(0, taille - TAILLE_MORCEAU)}):
            f.seek(debut)
            h.update(f.read(TAILLE_MORCEAU))
    return h.hexdigest()


def _chemin_registre():
    return os.path.join(dossier_cache(), "registre.json")


def _lire_registre():
    try:
        return json.load(open(_chemin_registre(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _ecrire_registre(reg):
    os.makedirs(dossier_cache(), exist_ok=True)
    tmp = _chemin_registre() + ".tmp"
    json.dump(reg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, _chemin_registre())        # écriture atomique : pas de registre à moitié écrit


def purger_expires(maintenant=None):
    """Supprime les entrées expirées ET leurs fichiers. Renvoie le nombre d'entrées supprimées."""
    maintenant = maintenant or time.time()
    reg = _lire_registre()
    expirees = [k for k, e in reg.items() if maintenant - e.get("cree_le", 0) > duree_conservation_s()]
    for k in expirees:
        d = reg[k].get("dossier")
        if d and os.path.isdir(d) and os.path.abspath(d).startswith(os.path.abspath(dossier_cache())):
            shutil.rmtree(d, ignore_errors=True)
        del reg[k]
    if expirees:
        _ecrire_registre(reg)
    return len(expirees)


def chercher(chemin_video, version=None):
    """Entrée réutilisable pour cette vidéo et cette version, ou None (vidéo inconnue, autre version ou expirée)."""
    purger_expires()
    e = _lire_registre().get(empreinte_video(chemin_video))
    if not e or e.get("version") != (version or version_courante()):
        return None
    if not all(os.path.exists(p) for p in e.get("fichiers", {}).values()):
        return None
    return e


def enregistrer(chemin_video, fichiers, version=None):
    """Copie les fichiers d'une analyse complète (ex. {"suivi": dossier du cache de suivi, "gemini": fichier .jsonl,
    "resultats": résumé .json}) dans le cache et les inscrit au registre. Renvoie l'entrée."""
    cle = empreinte_video(chemin_video)
    dossier = os.path.join(dossier_cache(), cle[:16])
    os.makedirs(dossier, exist_ok=True)
    copies = {}
    for nom, src in fichiers.items():
        dst = os.path.join(dossier, nom + ("" if os.path.isdir(src) else os.path.splitext(src)[1]))
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        copies[nom] = dst
    reg = _lire_registre()
    reg[cle] = {"version": version or version_courante(), "cree_le": time.time(), "dossier": dossier,
                "fichiers": copies, "nom_video": os.path.basename(chemin_video)}
    _ecrire_registre(reg)
    return reg[cle]
