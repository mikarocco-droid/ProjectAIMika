import anthropic
import subprocess
import base64
import json
import os
import re
import time
from dotenv import load_dotenv
from collections import Counter

load_dotenv(dotenv_path=r"D:\ProjetAIMika\.env\param.env")

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CLAUDE_API_KEY       = os.environ.get("CLAUDE_API_KEY", "")
BASE_OUTPUT_DIR      = "output"
FRAME_INTERVAL_LONG  = 30   # 1 frame/30s pour matchs > 20 min
FRAME_INTERVAL_SHORT = 5    # 1 frame/5s  pour vidéos  < 20 min
BATCH_SIZE           = 5    # frames par paquet envoyé à Claude
BATCH_DELAY          = 65   # secondes de pause entre paquets

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
print("CLAUDE KEY:", CLAUDE_API_KEY[:20] + "..." if CLAUDE_API_KEY else "NON TROUVÉE")

# ─────────────────────────────────────────
# DOSSIERS PAR VIDÉO
# ─────────────────────────────────────────
def setup_dirs(video_path: str) -> tuple:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    safe_name  = re.sub(r'[<>:"/\\|?*]', '_', video_name)
    output_dir = os.path.join(BASE_OUTPUT_DIR, safe_name)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(frames_dir):
        print("🗑️  Nettoyage des anciennes frames...")
        for f in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, f))
    os.makedirs(frames_dir, exist_ok=True)
    print(f"📁 Dossier : {output_dir}")
    return output_dir, frames_dir

# ─────────────────────────────────────────
# ÉTAPE 1 : Extraire les frames
# ─────────────────────────────────────────
def extract_frames(video_path: str, frames_dir: str) -> tuple:
    print("🎞️  Extraction des frames...")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result   = subprocess.run(cmd, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f"⏱️  Durée : {duration/60:.1f} minutes")

    interval = FRAME_INTERVAL_LONG if duration > 1200 else FRAME_INTERVAL_SHORT
    print(f"📸 Intervalle : 1 frame toutes les {interval}s")

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval},scale=640:-1", "-q:v", "5",
        os.path.join(frames_dir, "frame_%04d.jpg")
    ], capture_output=True)

    frames = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir) if f.endswith(".jpg")
    ])
    print(f"✅ {len(frames)} frames extraites")
    return frames, duration, interval

# ─────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────
def clean_json(text: str) -> str:
    text  = re.sub(r"```json|```", "", text).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("Pas de JSON trouvé")
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":   depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return text[start:]

def encode_frame(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def seconds_to_mmss(s: float) -> str:
    s = int(max(0, s))
    return f"{s//60:02d}:{s%60:02d}"

def time_to_seconds(t: str) -> float:
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0

# ─────────────────────────────────────────
# ÉTAPE 2 : Résumé global
# ─────────────────────────────────────────
def get_summary(actions: list, duration: float, sport: str = "football") -> dict:
    print("📝 Génération du résumé global...")
    prompt = f"""
    Voici les actions détectées dans un match de {sport} de {duration/60:.0f} minutes :
    {json.dumps(actions, ensure_ascii=False, indent=2)}

    Génère un rapport en JSON valide (sans markdown) :
    {{
        "resume": "Résumé du match en 3-4 phrases",
        "score": "Score estimé ou null",
        "highlights": [
            {{
                "timestamp_debut": "MM:SS",
                "timestamp_fin": "MM:SS",
                "type": "but|tir|dribble|passe_cle|action_defensive|autre",
                "description": "Description",
                "importance": 4
            }}
        ],
        "top_joueurs": [
            {{
                "joueur": "Description visuelle",
                "points_forts": "Ce qu'il fait bien",
                "note": 8
            }}
        ],
        "observations_tactiques": "2-3 observations clés"
    }}
    Garde les 10 highlights les plus importants par importance décroissante.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = clean_json(response.content[0].text)
    return json.loads(text)

# ─────────────────────────────────────────
# ÉTAPE 3 : Analyser les frames par paquets
# ─────────────────────────────────────────
def analyze_frames(frames: list, duration: float, interval: int,
                   sport: str = "football") -> dict:
    print("🤖 Analyse des frames avec Claude...")
    all_actions  = []
    total_frames = len(frames)

    for batch_start in range(0, total_frames, BATCH_SIZE):
        batch     = frames[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_b   = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  📦 Paquet {batch_num}/{total_b} ({len(batch)} frames)...")

        content = [{
            "type": "text",
            "text": (
                f"Tu es un expert scout de {sport}. "
                f"Voici {len(batch)} frames extraites d'un match, "
                f"1 frame toutes les {interval} secondes. "
                f"La 1ère frame = seconde {batch_start * interval}s du match.\n\n"
                "Pour chaque action importante détectée, indique :\n"
                "- frame : numéro dans CE paquet (commence à 1)\n"
                "- type : but|tir|dribble|passe_cle|action_defensive|faute|autre\n"
                "- description : courte\n"
                "- importance : entier 1-5\n\n"
                "Réponds UNIQUEMENT en JSON valide sans markdown :\n"
                '{"actions": [{"frame": 2, "type": "tir", "description": "Tir cadré", "importance": 3}]}'
            )
        }]
        for i, fp in enumerate(batch):
            content.append({"type": "text", "text": f"Frame {i+1} :"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": encode_frame(fp)
            }})

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=800,
                messages=[{"role": "user", "content": content}]
            )
            text    = clean_json(response.content[0].text)
            data    = json.loads(text)
            actions = data.get("actions", [])
            for action in actions:
                frame_abs = batch_start + action["frame"]
                ts_sec    = frame_abs * interval
                action["timestamp_sec"]   = ts_sec
                action["timestamp_debut"] = seconds_to_mmss(max(0, ts_sec - 5))
                action["timestamp_fin"]   = seconds_to_mmss(ts_sec + 10)
            all_actions.extend(actions)
            print(f"     → {len(actions)} action(s) détectée(s)")
        except Exception as e:
            print(f"     ⚠️  Erreur : {e}")

        if batch_start + BATCH_SIZE < total_frames:
            print(f"     ⏳ Pause {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)

    print(f"✅ Total : {len(all_actions)} actions détectées")
    return get_summary(all_actions, duration, sport)

# ─────────────────────────────────────────
# ÉTAPE 4 : Affiner les highlights
# ─────────────────────────────────────────
def refine_highlight(video_path: str, output_dir: str, timestamp_sec: float, label: str) -> dict:
    print(f"  🔍 Affinage : {label} autour de {seconds_to_mmss(int(timestamp_sec))}...")
    refine_dir = os.path.join(output_dir, "refine")
    os.makedirs(refine_dir, exist_ok=True)

    def clean_dir():
        for f in os.listdir(refine_dir):
            os.remove(os.path.join(refine_dir, f))

    # Passe 1 : 1 frame/5s sur ±60s
    clean_dir()
    start1 = max(0, timestamp_sec - 60)
    end1   = timestamp_sec + 60
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start1), "-to", str(end1),
        "-i", video_path, "-vf", "fps=0.2,scale=640:-1", "-q:v", "5",
        os.path.join(refine_dir, "p1_%04d.jpg")
    ], capture_output=True)

    frames1 = sorted([os.path.join(refine_dir, f)
                      for f in os.listdir(refine_dir) if f.startswith("p1_")])
    if not frames1:
        return None

    content1 = [{
        "type": "text",
        "text": (
            f"Ces frames couvrent {seconds_to_mmss(int(start1))} à {seconds_to_mmss(int(end1))} "
            f"(1 frame/5s). On cherche une action '{label}'.\n"
            f"Trouve la frame JUSTE AVANT que l'action commence.\n"
            f"JSON sans markdown : "
            f'{{"frame_debut": 3, "frame_fin": 6, "confiance": 4}}'
        )
    }]
    for i, fp in enumerate(frames1):
        ts = seconds_to_mmss(int(start1 + i * 5))
        content1.append({"type": "text",  "text": f"Frame {i+1} ({ts}) :"})
        content1.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": encode_frame(fp)
        }})

    time.sleep(BATCH_DELAY)
    try:
        r1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=200,
                                     messages=[{"role": "user", "content": content1}])
        d1 = json.loads(clean_json(r1.content[0].text))
        zone_start = start1 + max(0, d1["frame_debut"] - 1) * 5
        zone_end   = start1 + d1["frame_fin"] * 5
        print(f"     Passe 1 → {seconds_to_mmss(int(zone_start))} - {seconds_to_mmss(int(zone_end))}")
    except:
        zone_start = max(0, timestamp_sec - 20)
        zone_end   = timestamp_sec + 20

    # Passe 2 : 1 frame/1s sur la zone affinée
    clean_dir()
    start2 = max(0, zone_start - 5)
    end2   = zone_end + 5
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start2), "-to", str(end2),
        "-i", video_path, "-vf", "fps=1,scale=640:-1", "-q:v", "3",
        os.path.join(refine_dir, "p2_%04d.jpg")
    ], capture_output=True)

    frames2 = sorted([os.path.join(refine_dir, f)
                      for f in os.listdir(refine_dir) if f.startswith("p2_")])
    if not frames2:
        return {"timestamp_debut": seconds_to_mmss(int(zone_start)),
                "timestamp_fin":   seconds_to_mmss(int(zone_end)),
                "description": label, "confirmation": True}

    content2 = [{
        "type": "text",
        "text": (
            f"Ces frames couvrent {seconds_to_mmss(int(start2))} à {seconds_to_mmss(int(end2))} "
            f"(1 frame/seconde). On cherche '{label}'.\n"
            f"- Frame exacte où l'ACTION COMMENCE\n"
            f"- Frame où l'action SE TERMINE\n"
            f"Pour un but : commence à la dernière passe, finit à la célébration.\n"
            f"JSON sans markdown :\n"
            f'{{"frame_debut": 5, "frame_fin": 12, "description": "description précise", "confirmation": true}}'
        )
    }]
    for i, fp in enumerate(frames2):
        ts = seconds_to_mmss(int(start2 + i))
        content2.append({"type": "text",  "text": f"Frame {i+1} ({ts}) :"})
        content2.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg", "data": encode_frame(fp)
        }})

    time.sleep(BATCH_DELAY)
    try:
        r2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=300,
                                     messages=[{"role": "user", "content": content2}])
        d2 = json.loads(clean_json(r2.content[0].text))
        ts_debut = seconds_to_mmss(int(start2 + d2["frame_debut"] - 1))
        ts_fin   = seconds_to_mmss(int(start2 + d2["frame_fin"]))
        print(f"     Passe 2 → {ts_debut} - {ts_fin} ✅")
        return {"timestamp_debut": ts_debut, "timestamp_fin": ts_fin,
                "description": d2.get("description", label), "confirmation": True}
    except:
        return {"timestamp_debut": seconds_to_mmss(int(zone_start)),
                "timestamp_fin":   seconds_to_mmss(int(zone_end)),
                "description": label, "confirmation": True}

def refine_all_highlights(video_path: str, output_dir: str, analysis: dict) -> dict:
    print("\n🔬 Affinage des highlights importants...")
    for h in analysis.get("highlights", []):
        if h.get("importance", 0) >= 4:
            ts     = time_to_seconds(h["timestamp_debut"]) + 5
            result = refine_highlight(video_path, output_dir, ts, h["type"])
            if result and result.get("confirmation"):
                h["timestamp_debut"] = result["timestamp_debut"]
                h["timestamp_fin"]   = result["timestamp_fin"]
                h["description"]     = result.get("description", h["description"])
                print(f"  ✅ {h['type']} → {h['timestamp_debut']} - {h['timestamp_fin']}")
            else:
                print(f"  ⚠️  Pas pu affiner {h['type']}, estimation conservée")
    return analysis

# ─────────────────────────────────────────
# ÉTAPE 5 : Découper les highlights
# ─────────────────────────────────────────
def cut_highlights(video_path: str, output_dir: str, highlights: list) -> list:
    print("✂️  Découpe des highlights...")
    clips = []
    for i, h in enumerate(sorted(highlights, key=lambda h: h.get("importance", 1), reverse=True)):
        start = max(0, time_to_seconds(h["timestamp_debut"]))
        end   = time_to_seconds(h["timestamp_fin"])
        out   = os.path.join(output_dir, f"highlight_{i+1}_{h['type']}.mp4")
        r     = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", video_path, "-c", "copy", out],
            capture_output=True
        )
        if r.returncode == 0:
            clips.append({"fichier": out, "type": h["type"],
                           "description": h["description"], "importance": h.get("importance", 1)})
            print(f"  ✅ {h['description'][:50]}")
        else:
            print(f"  ❌ Erreur clip {i+1}")
    return clips

# ─────────────────────────────────────────
# ÉTAPE 6 : Highlight reel
# ─────────────────────────────────────────
def create_highlight_reel(output_dir: str, clips: list) -> str:
    if not clips:
        print("⚠️  Aucun clip")
        return ""
    print("🎬 Création du highlight reel...")
    list_path = os.path.join(output_dir, "clips_list.txt")
    with open(list_path, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c['fichier'])}'\n")
    reel = os.path.join(output_dir, "highlight_reel.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", list_path, "-c", "copy", reel], capture_output=True)
    print(f"✅ Reel : {reel}")
    return reel

# ─────────────────────────────────────────
# ÉTAPE 7 : Rapport match
# ─────────────────────────────────────────
def print_report(output_dir: str, analysis: dict, clips: list, reel_path: str):
    print("\n" + "═"*60)
    print("         📊 RAPPORT SCOUT IA")
    print("═"*60)
    print(f"\n📝 RÉSUMÉ\n{analysis.get('resume','')}")
    if analysis.get("score"):
        print(f"\n🏆 SCORE : {analysis['score']}")
    print(f"\n🎯 HIGHLIGHTS ({len(clips)} clips)")
    for i, c in enumerate(clips, 1):
        print(f"  {i}. [{c['type'].upper()}] {c['description']}")
        print(f"     {'⭐' * c['importance']}")
    if analysis.get("top_joueurs"):
        print("\n👤 TOP JOUEURS")
        for j in analysis["top_joueurs"]:
            print(f"  • {j['joueur']} — {j['note']}/10 — {j['points_forts']}")
    if analysis.get("observations_tactiques"):
        print(f"\n📈 TACTIQUE\n  {analysis['observations_tactiques']}")
    if reel_path:
        print(f"\n🎬 HIGHLIGHT REEL : {reel_path}")
    print("═"*60)
    with open(os.path.join(output_dir, "rapport.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print("💾 rapport.json sauvegardé\n")

# ─────────────────────────────────────────
# ANALYSE JOUEUR CIBLÉ
# ─────────────────────────────────────────
def analyze_player(frames: list, duration: float, interval: int,
                   numero: int, couleur: str,
                   position: str = "", sport: str = "football",
                   couleur_gardien_domicile: str = "",
                   couleur_gardien_visiteur: str = "") -> dict:

    print(f"🎯 Analyse #{numero} ({couleur}) — {position}...")

    gardien_context = ""
    if couleur_gardien_domicile:
        gardien_context += f"\n- Gardien équipe DOMICILE : vareuse {couleur_gardien_domicile}"
    if couleur_gardien_visiteur:
        gardien_context += f"\n- Gardien équipe VISITEUR : vareuse {couleur_gardien_visiteur}"
    if gardien_context:
        gardien_context = f"\n\nCOULEURS DES GARDIENS (ne pas confondre) :{gardien_context}"

    position_context = ""
    if position:
        is_gardien = "gardien" in position.lower()
        if is_gardien:
            equipe = "domicile" if "domicile" in position else "visiteur"
            position_context = (
                f"\nC'est le GARDIEN DE BUT de l'équipe {equipe}. "
                f"Focalise-toi sur ses interventions, sorties, relances et arrêts."
            )
        else:
            position_context = f"\nSa position est : {position}. Analyse ses actions typiques pour ce poste."

    all_actions  = []
    total_frames = len(frames)
    zones        = []

    for batch_start in range(0, total_frames, BATCH_SIZE):
        batch     = frames[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_b   = (total_frames + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  📦 Paquet {batch_num}/{total_b}...")

        content = [{
            "type": "text",
            "text": (
                f"Tu es un expert scout de {sport}. "
                f"Analyse ces frames et concentre-toi UNIQUEMENT sur le joueur "
                f"numéro {numero} avec un maillot {couleur}.{position_context}{gardien_context}\n"
                f"Frames : 1 toutes les {interval}s, début à {batch_start * interval}s.\n\n"
                f"Pour chaque frame où ce joueur est visible :\n"
                f"- frame : numéro dans ce paquet\n"
                f"- visible : true/false\n"
                f"- action : ce que fait le joueur\n"
                f"- zone : position sur le terrain\n"
                f"- avec_ballon : true/false\n"
                f"- evaluation : note 1-5\n\n"
                f"JSON sans markdown :\n"
                f'{{"observations": [{{"frame": 1, "visible": true, "action": "sprint", "zone": "milieu", "avec_ballon": false, "evaluation": 3}}]}}'
            )
        }]
        for i, fp in enumerate(batch):
            content.append({"type": "text",  "text": f"Frame {i+1} :"})
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": encode_frame(fp)
            }})

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1000,
                messages=[{"role": "user", "content": content}]
            )
            text = clean_json(response.content[0].text)
            obs  = json.loads(text).get("observations", [])
            for o in obs:
                if o.get("visible"):
                    ts = (batch_start + o["frame"]) * interval
                    o["timestamp"]     = seconds_to_mmss(ts)
                    o["timestamp_sec"] = ts
                    all_actions.append(o)
                    if o.get("zone"):
                        zones.append(o["zone"])
            print(f"     → {sum(1 for o in obs if o.get('visible'))} frame(s) avec joueur")
        except Exception as e:
            print(f"     ⚠️  Erreur : {e}")

        if batch_start + BATCH_SIZE < total_frames:
            print(f"     ⏳ Pause {BATCH_DELAY}s...")
            time.sleep(BATCH_DELAY)

    print(f"✅ {len(all_actions)} observations collectées")
    return generate_player_report(all_actions, zones, numero, couleur, sport, duration)


def generate_player_report(actions: list, zones: list, numero: int,
                            couleur: str, sport: str, duration: float) -> dict:
    print("📝 Génération du rapport joueur...")
    total_obs       = len(actions)
    avec_ballon     = sum(1 for a in actions if a.get("avec_ballon"))
    evaluations     = [a["evaluation"] for a in actions if a.get("evaluation")]
    note_moyenne    = round(sum(evaluations) / len(evaluations), 1) if evaluations else 0
    zone_counts     = Counter(zones)
    zone_principale = zone_counts.most_common(1)[0][0] if zone_counts else "inconnue"

    prompt = f"""
    Tu es un expert scout de {sport}.
    Observations du joueur #{numero} (maillot {couleur}) sur {duration/60:.0f} minutes :
    {json.dumps(actions, ensure_ascii=False, indent=2)}

    Stats : frames visibles={total_obs}, touches balle={avec_ballon},
            note moyenne={note_moyenne}/5, zone principale={zone_principale}

    Génère un rapport en JSON valide (sans markdown) :
    {{
        "joueur": "#{numero} - maillot {couleur}",
        "note_globale": 7,
        "resume": "Résumé en 3-4 phrases",
        "points_forts": ["point 1", "point 2", "point 3"],
        "points_faibles": ["point 1", "point 2"],
        "stats": {{
            "touches_balle": {avec_ballon},
            "zone_principale": "{zone_principale}",
            "note_moyenne_actions": {note_moyenne},
            "frames_visibles": {total_obs}
        }},
        "moments_cles": [
            {{"timestamp": "MM:SS", "description": "Action remarquable", "evaluation": 4}}
        ],
        "recommandation": "Recommandation pour le coach"
    }}
    """
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = clean_json(response.content[0].text)
    return json.loads(text)


def print_player_report(report: dict):
    print("\n" + "═"*60)
    print(f"       🎯 RAPPORT SCOUT — {report.get('joueur','')}")
    print("═"*60)
    print(f"\n⭐ NOTE GLOBALE : {report.get('note_globale','')}/10")
    print(f"\n📝 RÉSUMÉ\n{report.get('resume','')}")
    if report.get("points_forts"):
        print("\n✅ POINTS FORTS")
        for p in report["points_forts"]:
            print(f"  • {p}")
    if report.get("points_faibles"):
        print("\n⚠️  À AMÉLIORER")
        for p in report["points_faibles"]:
            print(f"  • {p}")
    if report.get("stats"):
        s = report["stats"]
        print(f"\n📊 STATS")
        print(f"  • Touches : {s.get('touches_balle','')}")
        print(f"  • Zone    : {s.get('zone_principale','')}")
        print(f"  • Note moy: {s.get('note_moyenne_actions','')}/5")
    if report.get("moments_cles"):
        print(f"\n🎬 MOMENTS CLÉS")
        for m in report["moments_cles"]:
            print(f"  [{m['timestamp']}] ⭐{m['evaluation']}/5 — {m['description']}")
    if report.get("recommandation"):
        print(f"\n💡 RECOMMANDATION\n  {report['recommandation']}")
    print("═"*60)

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH      = r"D:\Video\Match Bullange - Stavelot B 0-1.mp4"
    SPORT           = "football"

    ANALYSE_MATCH   = True
    ANALYSE_JOUEUR  = False
    NUMERO_JOUEUR   = 9
    COULEUR_MAILLOT = "bordeaux"
    POSITION_JOUEUR = "avant-centre"

    output_dir, frames_dir     = setup_dirs(VIDEO_PATH)
    frames, duration, interval = extract_frames(VIDEO_PATH, frames_dir)

    if ANALYSE_MATCH:
        analysis = analyze_frames(frames, duration, interval, sport=SPORT)
        analysis = refine_all_highlights(VIDEO_PATH, output_dir, analysis)
        clips    = cut_highlights(VIDEO_PATH, output_dir, analysis.get("highlights", []))
        reel     = create_highlight_reel(output_dir, clips)
        print_report(output_dir, analysis, clips, reel)

    if ANALYSE_JOUEUR:
        player_report = analyze_player(
            frames, duration, interval,
            numero=NUMERO_JOUEUR, couleur=COULEUR_MAILLOT,
            position=POSITION_JOUEUR, sport=SPORT,
        )
        print_player_report(player_report)
        path = os.path.join(output_dir, f"rapport_joueur_{NUMERO_JOUEUR}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(player_report, f, ensure_ascii=False, indent=2)
        print(f"💾 Rapport joueur : {path}")