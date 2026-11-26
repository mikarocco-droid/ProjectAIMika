# analyse_execution.py — 26/11/2026
# LE CALCUL d'une analyse, sans base de données : utilisé à l'identique
#   • par le site (app.run_analysis), quand l'analyse tourne sur la machine du site ;
#   • par la machine GPU distante (worker_gpu.py), en production.
# Reprend tel quel le contenu validé de run_analysis (vitesse/Flex, cache Gemini par analyse, réutilisation d'un match
# déjà analysé, type de caméra, analyse incomplète refusée, mise en cache, empreinte de la vidéo).
import os
from datetime import datetime, timedelta

import config


def executer_analyse(analysis_id, video_path, sport, plan, output_dir, vitesse="rapide", cree_le=None,
                     mode="match", player_id=None, team_names=None, player_position=None, kickoff_s_precalcule=None,
                     team_colors_gemini_precalcule=None, use_match_boundary_segments=True,
                     couleur_maillot_gardien=None, **_ignores):
    """Fait l'analyse complète. Renvoie (result, empreinte). Lève une exception si elle n'aboutit pas."""
    from pipeline import run_pipeline
    try:
        from ai.gemini_validator import gemini_reinitialiser_etat
        gemini_reinitialiser_etat()
    except Exception as _e_gv:
        print(f"  ⚠️  Remise à zéro de l'état Gemini impossible : {_e_gv}")

    # VITESSE — « 48h » = Gemini Flex (−50 %), à finir 36 h après la commande ; « rapide » = tarif standard
    cree_le = cree_le or datetime.utcnow()
    if vitesse == "48h":
        os.environ["SCOUTIA_GEMINI_TIER"] = "flex"
        os.environ["SCOUTIA_FLEX_ECHEANCE"] = str((cree_le + timedelta(hours=36) - datetime(1970, 1, 1)).total_seconds())
    else:
        os.environ["SCOUTIA_GEMINI_TIER"] = ""
        os.environ.pop("SCOUTIA_FLEX_ECHEANCE", None)

    # RÉUTILISATION d'une analyse du même match (même fichier, même version des réglages)
    _match_data_reutilise, _entree_cache = None, None
    os.makedirs(output_dir, exist_ok=True)
    os.environ["SCOUTIA_GEMINI_CACHE"] = "on"
    os.environ["SCOUTIA_GEMINI_CACHE_OUT"] = os.path.join(output_dir, "gemini_cache.jsonl")
    os.environ["SCOUTIA_GEMINI_CACHE_VIDEO"] = os.path.basename(video_path)
    os.environ["SCOUTIA_GEMINI_CACHE_IN"] = os.path.join(output_dir, "aucun_cache.jsonl")
    os.environ.pop("SCOUTIA_CACHE_JOUEUR", None)
    try:
        from analysis.cache_analyses import chercher as _chercher_cache
        _entree_cache = _chercher_cache(video_path)
        if _entree_cache:
            from replay import load_cache as _load_cache_r
            _match_data_reutilise = _load_cache_r(_entree_cache["fichiers"]["suivi"])
            os.environ["SCOUTIA_GEMINI_CACHE_IN"] = _entree_cache["fichiers"]["gemini"]
            os.environ["SCOUTIA_CACHE_JOUEUR"] = _entree_cache["fichiers"]["suivi"]
            print(f"  ♻️  Match déjà analysé ({_entree_cache.get('nom_video')}) : suivi et réponses Gemini réutilisés")
    except Exception as _e_cache:
        print(f"  ⚠️  Réutilisation impossible, analyse complète : {_e_cache}")
        _match_data_reutilise, _entree_cache = None, None
    try:
        from ai.gemini_validator import gemini_cache_reinitialiser
        gemini_cache_reinitialiser()
    except Exception:
        pass

    result = run_pipeline(
        video_path=video_path, sport=sport, output_dir=output_dir, analysis_id=analysis_id,
        save_annotated=config.PLANS.get(plan, {}).get("montage", False), plan=plan, mode=mode,
        player_id=player_id, player_position=player_position, team_names=team_names or None,
        kickoff_s_precalcule=kickoff_s_precalcule, team_colors_gemini_precalcule=team_colors_gemini_precalcule,
        use_match_boundary_segments=use_match_boundary_segments, couleur_maillot_gardien=couleur_maillot_gardien,
        _match_data=_match_data_reutilise,
        camera_type=os.getenv("SCOUTIA_CAMERA_TYPE", "high_side"),    # même type que toute la validation
    )
    if isinstance(result, dict):
        result["analyse_reutilisee"] = bool(_entree_cache)

    # une analyse INCOMPLÈTE (questions au service d'IA abandonnées : crédit, quota) n'est pas livrée comme réussie
    try:
        from ai.gemini_validator import gemini_etat
        _etat_gv = gemini_etat()
    except Exception:
        _etat_gv = {}
    if _etat_gv.get("questions_abandonnees"):
        raise RuntimeError(f"Analyse incomplète : {_etat_gv['questions_abandonnees']} question(s) au service d'IA "
                           f"n'ont pas abouti (crédit ou quota) — à relancer")

    # analyse complète → mise en cache pour réutilisation (suivi + réponses Gemini)
    if not _entree_cache:
        try:
            from analysis.cache_analyses import enregistrer as _enregistrer_cache
            _suivi = os.path.join(output_dir, "cache")
            _gem = os.environ.get("SCOUTIA_GEMINI_CACHE_OUT")
            if os.path.exists(os.path.join(_suivi, "frames_data.pkl")) and _gem and os.path.exists(_gem):
                _enregistrer_cache(video_path, {"suivi": _suivi, "gemini": _gem})
                print("  ♻️  Analyse mise en cache : le même match ne sera plus réanalysé")
        except Exception as _e_enr:
            print(f"  ⚠️  Mise en cache impossible : {_e_enr}")

    # empreinte du contenu (corrections de buts partagées entre utilisateurs de la même vidéo)
    _empreinte = None
    try:
        from analysis.cache_analyses import empreinte_video
        _empreinte = empreinte_video(video_path)
    except Exception as _e_emp:
        print(f"  ⚠️  Empreinte de la vidéo impossible : {_e_emp}")
    return result, _empreinte
