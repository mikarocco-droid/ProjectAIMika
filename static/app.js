// static/app.js — ScoutIA V9.7

// ─────────────────────────────────────────
// ÉLÉMENTS
// ─────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const fileName       = document.getElementById("file-name");
const submitBtn      = document.getElementById("submit-btn");
const browseBtn      = document.getElementById("browse-btn");
const sportSelect    = document.getElementById("sport-select");
const modeMatch      = document.getElementById("mode-match");
const modeJoueur     = document.getElementById("mode-joueur");
const joueurOpts     = document.getElementById("joueur-options");
const numJoueur      = document.getElementById("numero-joueur");
const positionEl     = document.getElementById("position-joueur");
const gardienSection = document.getElementById("gardien-section");
const fieldNumero    = document.getElementById("field-numero");
const matchDesc      = document.getElementById("mode-match-desc");
const joueurDesc     = document.getElementById("mode-joueur-desc");
const stepConfig     = document.getElementById("step-config");
const progressWrap   = document.getElementById("upload-progress-wrap");
const progressBar    = document.getElementById("upload-bar");
const progressPct    = document.getElementById("upload-pct");
const progressSize   = document.getElementById("upload-size");
const uploadDoneMsg  = document.getElementById("upload-done-msg");
const previewUploadId = document.getElementById("preview-upload-id");

// ─────────────────────────────────────────
// CONFIG SPORTS
// ─────────────────────────────────────────
const SPORTS_CONFIG = {
    football: {
        icon: "⚽", hasGardien: true, hasNumero: true,
        matchDesc:  "Buts, occasions, résumé vidéo et PDF",
        joueurDesc: "Montage vidéo d'un joueur (moments où son numéro est lisible)",
        positions: [
            { group: "🥅 Gardien",  opts: ["gardien|Gardien"] },
            { group: "🛡️ Défense",  opts: ["defenseur|Défenseur", "lateraldroit|Latéral droit", "lateralgauche|Latéral gauche"] },
            { group: "⚙️ Milieu",   opts: ["milieu|Milieu", "milieud|Milieu défensif", "milieuof|Milieu offensif", "ailier|Ailier"] },
            { group: "⚡ Attaque",  opts: ["attaquant|Attaquant", "avant-centre|Avant-centre"] },
        ]
    },
    "mini-foot": {
        icon: "⚽", hasGardien: true, hasNumero: true,
        matchDesc:  "Analyse mini-foot / futsal",
        joueurDesc: "Rapport joueur futsal",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien|Gardien"] },
            { group: "⚡ Joueurs", opts: ["pivot|Pivot", "ala|Ala", "fixo|Fixo"] },
        ]
    },
    basketball: {
        icon: "🏀", hasGardien: false, hasNumero: true,
        matchDesc:  "Stats basket, highlights, rapport",
        joueurDesc: "Rapport poste basket",
        positions: [
            { group: "🏀 Postes", opts: ["pg|Meneur (PG)", "sg|Arrière (SG)", "sf|Ailier (SF)", "pf|Ailier fort (PF)", "c|Pivot (C)"] }
        ]
    },
    handball: {
        icon: "🤾", hasGardien: true, hasNumero: true,
        matchDesc:  "Analyse handball complète",
        joueurDesc: "Rapport joueur handball",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien|Gardien"] },
            { group: "⚡ Joueurs", opts: ["pivot|Pivot", "ailier|Ailier", "arrieredroit|Arrière droit", "arrierecentre|Demi-centre", "arrierecentreg|Arrière gauche"] },
        ]
    },
    tennis: {
        icon: "🎾", hasGardien: false, hasNumero: false,
        matchDesc:  "Analyse tennis — stats, highlights",
        joueurDesc: "Rapport joueur tennis",
        positions: []
    },
};


// ─────────────────────────────────────────
// UPLOAD EN 2 ÉTAPES
// ─────────────────────────────────────────
let _uploadDone = false;

// ─────────────────────────────────────────
// ENVOI + DÉTECTION PENDANT L'ENVOI (24/10/2026)
// Le coup d'envoi et les couleurs n'ont besoin que du DÉBUT de la vidéo : un aperçu (12 premières minutes + index
// MP4) est envoyé EN PARALLÈLE du fichier complet, et la configuration (noms, couleurs, coup d'envoi) s'affiche dès
// que la détection est finie. Le bouton « Lancer l'analyse » attend la fin du téléchargement complet.
// Si l'aperçu est impossible (format non MP4/MOV, erreur), détection sur le fichier complet comme avant.
// ─────────────────────────────────────────
const DUREE_APERCU_S = 720;            // 12 min : couvre le coup d'envoi de la plupart des vidéos
let _detectionLancee = false;          // une détection (aperçu ou fichier complet) est en cours ou faite
let _apercuEnCours   = false;          // l'aperçu est en préparation ou en cours d'envoi
let _uidComplet      = "";

function revealConfig() {
    const stepDetecting = document.getElementById("step-detecting");
    if (stepDetecting) stepDetecting.style.display = "none";
    if (stepConfig && stepConfig.style.display !== "block") {
        stepConfig.style.display = "block";
        setTimeout(() => stepConfig.scrollIntoView({ behavior: "smooth", block: "start" }), 150);
    }
    updateSubmitBtn();
}

function afficherDetectionEnCours() {
    const stepDetecting = document.getElementById("step-detecting");
    if (stepDetecting) {
        stepDetecting.style.display = "block";
        stepDetecting.scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

async function _lireOctets(file, debut, n) {
    return new DataView(await file.slice(debut, Math.min(file.size, debut + n)).arrayBuffer());
}

async function lireBoites(file) {
    // Boîtes de premier niveau d'un MP4/MOV : [taille 32 bits][type 4 octets] (+ taille 64 bits si taille == 1)
    const boites = [];
    let o = 0;
    while (o + 8 <= file.size && boites.length < 64) {
        const dv = await _lireOctets(file, o, 16);
        let taille = dv.getUint32(0);
        const type = String.fromCharCode(dv.getUint8(4), dv.getUint8(5), dv.getUint8(6), dv.getUint8(7));
        let entete = 8;
        if (taille === 1 && dv.byteLength >= 16) { taille = Number(dv.getBigUint64(8)); entete = 16; }
        else if (taille === 0) { taille = file.size - o; }
        if (taille < entete) break;                      // fichier non MP4 ou corrompu
        boites.push({ type, offset: o, taille, entete });
        o += taille;
    }
    return boites;
}

function _enfants(dv, debut, fin) {
    // Boîtes enfants d'une boîte déjà lue en mémoire (dv = DataView du parent)
    const out = [];
    let o = debut;
    while (o + 8 <= fin) {
        const t = dv.getUint32(o);
        const type = String.fromCharCode(dv.getUint8(o + 4), dv.getUint8(o + 5), dv.getUint8(o + 6), dv.getUint8(o + 7));
        if (t < 8 || o + t > fin) break;
        out.push({ type, debut: o, fin: o + t });
        o += t;
    }
    return out;
}

async function _apercuFragmente(file, boites, moov) {
    // 24/10/2026 : MP4 FRAGMENTÉ (caméras de longue durée, ex. Veo) — chaque fragment (moof + mdat) a son propre
    // index : le DÉBUT du fichier, tel quel, est lisible. Le débit (octets par seconde) est estimé à partir de
    // l'heure de début (« tfdt ») du premier et du dernier fragment lus.
    const moofs = boites.filter(b => b.type === "moof");
    if (moofs.length < 2 || moov.taille > 4 * 1024 * 1024) return null;
    const mv = await _lireOctets(file, moov.offset, moov.taille);
    const echelles = {};                               // track_ID -> échelle de temps (mdhd)
    for (const trak of _enfants(mv, 8, mv.byteLength).filter(b => b.type === "trak")) {
        let id = null, ech = null;
        for (const e of _enfants(mv, trak.debut + 8, trak.fin)) {
            if (e.type === "tkhd") { const v = mv.getUint8(e.debut + 8); id = mv.getUint32(e.debut + (v === 1 ? 28 : 20)); }
            if (e.type === "mdia") {
                for (const m of _enfants(mv, e.debut + 8, e.fin)) {
                    if (m.type === "mdhd") { const v = mv.getUint8(m.debut + 8); ech = mv.getUint32(m.debut + (v === 1 ? 28 : 20)); }
                }
            }
        }
        if (id !== null && ech) echelles[id] = ech;
    }
    async function heureFragment(moof) {
        const dv = await _lireOctets(file, moof.offset, moof.taille);
        for (const traf of _enfants(dv, 8, dv.byteLength).filter(b => b.type === "traf")) {
            let id = null, t = null;
            for (const e of _enfants(dv, traf.debut + 8, traf.fin)) {
                if (e.type === "tfhd") id = dv.getUint32(e.debut + 12);
                if (e.type === "tfdt") { const v = dv.getUint8(e.debut + 8); t = v === 1 ? Number(dv.getBigUint64(e.debut + 12)) : dv.getUint32(e.debut + 12); }
            }
            if (id !== null && t !== null && echelles[id]) return t / echelles[id];
        }
        return null;
    }
    const t0 = await heureFragment(moofs[0]), t1 = await heureFragment(moofs[moofs.length - 1]);
    if (t0 === null || t1 === null || t1 <= t0) return null;
    const debit = (moofs[moofs.length - 1].offset - moofs[0].offset) / (t1 - t0);
    const L = Math.min(file.size, Math.round(moofs[0].offset + debit * DUREE_APERCU_S));
    const duree_apercu_s = Math.min(DUREE_APERCU_S, (L - moofs[0].offset) / debit);
    return { L, fragmente: true, duree_apercu_s };
}

async function preparerApercu(file) {
    if (!/\.(mp4|mov|m4v)$/i.test(file.name || "")) return null;
    const boites = await lireBoites(file);
    const moov = boites.find(b => b.type === "moov"), mdat = boites.find(b => b.type === "mdat");
    if (!moov) return null;
    if (boites.some(b => b.type === "moof")) return _apercuFragmente(file, boites, moov);
    if (!mdat) return null;
    // Durée de la vidéo (boîte « mvhd », au début de « moov »)
    const tete = await _lireOctets(file, moov.offset, 256);
    let p = -1;
    for (let i = 0; i + 4 <= tete.byteLength; i++) {
        if (tete.getUint8(i) === 0x6d && tete.getUint8(i + 1) === 0x76 && tete.getUint8(i + 2) === 0x68 && tete.getUint8(i + 3) === 0x64) { p = i; break; }
    }
    if (p < 0) return null;
    const v = tete.getUint8(p + 4);
    const echelle = v === 1 ? tete.getUint32(p + 24) : tete.getUint32(p + 16);
    const duree   = v === 1 ? Number(tete.getBigUint64(p + 28)) : tete.getUint32(p + 20);
    const duree_s = echelle ? duree / echelle : 0;
    if (!duree_s) return null;
    const duree_apercu_s = Math.min(DUREE_APERCU_S, duree_s);
    const L = Math.min(file.size, Math.round(mdat.offset + mdat.entete + mdat.taille / duree_s * duree_apercu_s));
    return { L, moov, mdat, duree_apercu_s };
}

async function lancerApercu(file) {
    _apercuEnCours = true;
    try {
        const ap = await preparerApercu(file);
        if (!ap || _detectionLancee) return;
        const fd = new FormData();
        fd.append("debut", file.slice(0, ap.L), "debut.mp4");
        if (!ap.fragmente) {               // MP4 classique : index et taille du bloc d'images à corriger
            if (ap.moov.offset >= ap.L) fd.append("index", file.slice(ap.moov.offset, ap.moov.offset + ap.moov.taille), "index.bin");
            fd.append("mdat_offset", ap.mdat.offset);
            fd.append("mdat_entete", ap.mdat.entete);
            fd.append("index_offset", ap.moov.offset);
        }
        fd.append("duree_apercu_s", ap.duree_apercu_s);
        const resp = await fetch("/api/apercu-debut", { method: "POST", body: fd });
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();
        if (!data.upload_id || _detectionLancee) return;
        _detectionLancee = true;
        afficherDetectionEnCours();
        await runTeamDetection(data.upload_id);
        revealConfig();
    } catch (e) {
        console.warn("Aperçu pendant l'envoi indisponible :", e.message);
    } finally {
        _apercuEnCours = false;
        // L'aperçu n'a pas servi et le fichier complet est déjà arrivé : détection sur le fichier complet
        if (!_detectionLancee && _uploadDone) detecterSurFichierComplet();
    }
}

function detecterSurFichierComplet() {
    if (_detectionLancee) return;
    _detectionLancee = true;
    afficherDetectionEnCours();
    if (_uidComplet) runTeamDetection(_uidComplet).then(revealConfig).catch(revealConfig);
    else revealConfig();
}

function startUploadWithProgress(file) {
    if (!file) return;
    _uploadDone = false; _detectionLancee = false; _uidComplet = "";

    // Afficher progression
    if (progressWrap) progressWrap.style.display = "block";
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    if (progressSize) progressSize.textContent = sizeMB + " MB";
    if (fileName) {
        fileName.textContent = file.name + "  (" + sizeMB + " MB)";
        fileName.style.color = "var(--cyan)";
    }

    const fd  = new FormData();
    fd.append("video", file);
    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (e) => {
        if (!e.lengthComputable) return;
        const p = Math.round(e.loaded / e.total * 100);
        if (progressBar) progressBar.style.width = p + "%";
        if (progressPct) progressPct.textContent = p + "%";
        if (submitBtn && !_uploadDone) submitBtn.textContent = "Téléchargement en cours… " + p + "%";
    };

    xhr.onload = () => {
        if (progressWrap) progressWrap.style.display = "none";
        if (xhr.status !== 200) {
            let msg = "Erreur " + xhr.status;
            try { msg = JSON.parse(xhr.responseText).message || msg; } catch (e) {}
            showUploadError(msg);
            return;
        }
        try {
            const data = JSON.parse(xhr.responseText);
            _uidComplet = data.upload_id || "";
            if (previewUploadId) previewUploadId.value = _uidComplet;   // c'est ce fichier qui sera analysé
            _uploadDone = true;
            updateSubmitBtn();
            // Détection sur le fichier complet seulement si l'aperçu n'a pas pu servir
            if (!_detectionLancee && !_apercuEnCours) detecterSurFichierComplet();
        } catch (e) {
            showUploadError("Réponse inattendue du serveur");
        }
    };

    xhr.onerror = () => showUploadError("Erreur réseau");

    xhr.open("POST", "/api/upload-preview");
    xhr.send(fd);

    // En parallèle : aperçu du début de la vidéo, pour la détection pendant l'envoi
    lancerApercu(file);
}

function showUploadError(msg) {
    if (progressWrap) {
        progressWrap.innerHTML =
            `<div style="color:var(--red);font-size:0.82rem">❌ ${msg} — réessayez</div>`;
    }
    _pickerLocked = false;
}


// ─────────────────────────────────────────
// DÉTECTION ÉQUIPES
// ─────────────────────────────────────────
async function runTeamDetection(uploadId) {
    const prog = document.getElementById("detecting-progress");

    // Timeout de sécurité : max 90s, puis on révèle quand même le dashboard
    const TIMEOUT_MS = 90000;
    let   timedOut   = false;
    const timeoutId  = setTimeout(() => {
        timedOut = true;
    }, TIMEOUT_MS);

    // Messages de progression pendant l'attente
    const msgs = [
        "Lecture des premières minutes de la vidéo…",
        "Détection des joueurs en cours…",
        "Analyse des couleurs de maillot…",
        "Calcul des clusters d'équipes…",
    ];
    let msgIdx = 0;
    const msgInterval = setInterval(() => {
        if (prog && msgIdx < msgs.length) {
            prog.textContent = msgs[msgIdx++];
        }
    }, 6000);

    try {
        if (prog) prog.textContent = msgs[0];

        // Fetch avec AbortController pour pouvoir annuler
        const controller = new AbortController();
        const fetchPromise = fetch(`/api/detect-teams/${uploadId}`,
                                   { signal: controller.signal });

        // Race entre la réponse et le timeout
        const resp = await Promise.race([
            fetchPromise,
            new Promise((_, reject) =>
                setTimeout(() => {
                    controller.abort();
                    reject(new Error("timeout"));
                }, TIMEOUT_MS)
            )
        ]);

        clearInterval(msgInterval);
        clearTimeout(timeoutId);

        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const data = await resp.json();

        if (!data.success) {
            if (prog) prog.textContent = data.error || "Détection non disponible — renseignez manuellement.";
            afficherKickoff(data.kickoff);         // saisie manuelle facultative du coup d'envoi
            await new Promise(r => setTimeout(r, 1500));
            return;
        }

        if (prog) prog.textContent = "✅ Couleurs détectées !";

        // Mettre à jour les dots + labels + placeholders
        for (const tid of [0, 1]) {
            const team = data["team_" + tid];
            if (!team) continue;

            const dot = document.getElementById(`team${tid}-color-dot`);
            if (dot && team.color_hex) dot.style.background = team.color_hex;

            const lbl = document.getElementById(`team${tid}-color-label`);
            if (lbl && team.color_name) lbl.textContent = "(" + team.color_name + ")";

            const inp = document.getElementById(`team-name-input-${tid}`);
            if (inp && !inp.value && team.color_name)
                inp.placeholder = "Ex: Équipe " + team.color_name;

            const opt = document.getElementById(`team-side-opt-${tid}`);
            if (opt && team.color_name)
                opt.textContent = (tid === 0 ? "🏠 " : "✈️ ") + "Équipe " + team.color_name;

            // 24/10/2026 : couleurs transmises au pipeline (champs cachés prévus par le dashboard, jamais remplis :
            // le pipeline ne pouvait pas rattacher les noms saisis aux équipes, ni le mode joueur à la couleur)
            const col = document.getElementById(`team-color-${tid}-input`);
            if (col) col.value = team.color_name || "";
            const conf = document.getElementById(`team-confidence-${tid}-input`);
            if (conf) conf.value = (team.confidence ?? team.confiance ?? "");
        }

        // 24/10/2026 : coup d'envoi de la pré-analyse (renvoyé par le serveur, jamais exploité : le pipeline le
        // recherchait une seconde fois)
        afficherKickoff(data.kickoff);

        await new Promise(r => setTimeout(r, 800));

    } catch (e) {
        clearInterval(msgInterval);
        clearTimeout(timeoutId);

        const msg = e.message === "timeout"
            ? "Délai dépassé — renseignez les couleurs manuellement."
            : "Détection non disponible — continuez manuellement.";
        if (prog) prog.textContent = msg;
        await new Promise(r => setTimeout(r, 1500));
        console.warn("Team detection:", e.message);
    }
}


// ─────────────────────────────────────────
// FILE HANDLING — UN SEUL ENDROIT
// ─────────────────────────────────────────
if (dropZone) {
    // Clic browse-btn ou drop-zone → ouvrir explorateur
    if (browseBtn) {
        browseBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }
    dropZone.addEventListener("click", (e) => {
        if (e.target === browseBtn || browseBtn?.contains(e.target)) return;
        fileInput.click();
    });

    // Drag & drop
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && fileInput) {
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            startUploadWithProgress(file);
        }
    });

    // Changement fichier
    fileInput?.addEventListener("change", () => {
        _pickerLocked = false;
        const file = fileInput.files[0];
        if (file) startUploadWithProgress(file);
    });
}


// ─────────────────────────────────────────
// SYNC NOMS ÉQUIPES → SELECT JOUEUR
// ─────────────────────────────────────────
for (const tid of [0, 1]) {
    const inp = document.getElementById(`team-name-input-${tid}`);
    const opt = document.getElementById(`team-side-opt-${tid}`);
    if (!inp || !opt) continue;
    inp.addEventListener("input", () => {
        const val  = inp.value.trim();
        const icon = tid === 0 ? "🏠 " : "✈️ ";
        opt.textContent = val ? icon + val : icon + "Équipe " + (tid === 0 ? "A" : "B");
    });
}


// ─────────────────────────────────────────
// POSITIONS DYNAMIQUES
// ─────────────────────────────────────────
function buildPositions(sport) {
    const cfg = SPORTS_CONFIG[sport];
    if (!cfg || !positionEl) return;
    positionEl.innerHTML = '<option value="">-- Choisir --</option>';
    cfg.positions.forEach(group => {
        const og = document.createElement("optgroup");
        og.label = group.group;
        group.opts.forEach(o => {
            const [val, label] = o.split("|");
            const opt = document.createElement("option");
            opt.value = val; opt.textContent = label;
            og.appendChild(opt);
        });
        positionEl.appendChild(og);
    });
}


// ─────────────────────────────────────────
// UPDATE SPORT UI
// ─────────────────────────────────────────
function updateSport() {
    const sport = sportSelect?.value || "football";
    const cfg   = SPORTS_CONFIG[sport] || SPORTS_CONFIG.football;
    if (matchDesc)      matchDesc.textContent  = cfg.matchDesc;
    if (joueurDesc)     joueurDesc.textContent = cfg.joueurDesc;
    if (fieldNumero)    fieldNumero.style.display    = cfg.hasNumero  ? "block" : "none";
    if (gardienSection) gardienSection.style.display = cfg.hasGardien ? "block" : "none";
    buildPositions(sport);
    updateSubmitBtn();
}


// ─────────────────────────────────────────
// MODE JOUEUR
// ─────────────────────────────────────────
function toggleJoueurOptions() {
    const isJoueur = modeJoueur?.checked;
    if (joueurOpts) joueurOpts.style.display = isJoueur ? "block" : "none";
    updateSubmitBtn();
}

modeMatch?.addEventListener("change",  toggleJoueurOptions);
modeJoueur?.addEventListener("change", toggleJoueurOptions);
sportSelect?.addEventListener("change", updateSport);
numJoueur?.addEventListener("input",   updateSubmitBtn);
positionEl?.addEventListener("change", updateSubmitBtn);


// ─────────────────────────────────────────
// VALIDATION SUBMIT
// ─────────────────────────────────────────
function updateSubmitBtn() {
    if (!submitBtn) return;
    const sport    = sportSelect?.value || "football";
    const cfg      = SPORTS_CONFIG[sport] || SPORTS_CONFIG.football;
    const isJoueur = modeJoueur?.checked;

    if (!_uploadDone) {
        submitBtn.disabled    = true;
        if (!/Téléchargement en cours/.test(submitBtn.textContent)) submitBtn.textContent = "Sélectionnez une vidéo";
        return;
    }

    if (isJoueur) {
        const hasNum = !cfg.hasNumero || numJoueur?.value;
        const hasPos = positionEl?.value;
        if (!hasNum || !hasPos) {
            submitBtn.disabled    = true;
            submitBtn.textContent = "Complétez les infos joueur";
            return;
        }
        submitBtn.disabled    = false;
        submitBtn.textContent = "Analyser le joueur " + cfg.icon;
        return;
    }

    submitBtn.disabled    = false;
    submitBtn.textContent = "Lancer l'analyse " + cfg.icon;
}


// ─────────────────────────────────────────
// FORM SUBMIT
// ─────────────────────────────────────────
const uploadForm = document.getElementById("upload-form");
if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
        if (submitBtn) {
            submitBtn.disabled    = true;
            submitBtn.textContent = "Envoi en cours...";
            submitBtn.style.opacity = "0.6";
        }
    });
}


// ─────────────────────────────────────────
// POLLING PROGRESSION
// ─────────────────────────────────────────
const POLL_INTERVAL = 3000;

document.querySelectorAll(".progress-container[data-id]").forEach(el => {
    poll(el.dataset.id, el);
});

async function poll(id, el) {
    await sleep(POLL_INTERVAL);
    try {
        const res  = await fetch(`/api/status/${id}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        const fill = el.querySelector(".progress-fill");
        const msg  = el.querySelector(".progress-msg");
        const pct  = Math.max(0, Math.min(100, data.progress || 0));

        if (fill) { fill.style.width = pct + "%"; fill.style.transition = "width 0.5s ease"; }
        if (msg) {
            if (data.status === "error") {
                msg.textContent = "Erreur : " + (data.progress_msg || "inconnue");
                msg.style.color = "var(--red)";
            } else {
                msg.textContent = data.progress_msg || `Analyse en cours... ${pct}%`;
                msg.style.color = "";
            }
        }

        if (data.status === "done") {
            if (fill) { fill.style.width = "100%"; fill.style.background = "var(--green)"; }
            if (msg)  { msg.textContent = "Terminé !"; msg.style.color = "var(--green)"; }
            setTimeout(() => location.reload(), 800);
        } else if (data.status !== "error") {
            poll(id, el);
        }

    } catch (err) {
        console.warn(`Poll ${id}:`, err.message);
        poll(id, el);
    }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }


// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────
updateSport();
toggleJoueurOptions();


// ─────────────────────────────────────────
// COUP D'ENVOI ET ÉQUIPE (24/10/2026) — fonctions appelées par le dashboard, absentes jusqu'ici
// ─────────────────────────────────────────
function _mmss(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
}

function afficherKickoff(k) {
    const input  = document.getElementById("kickoff-s-input");
    const auto   = document.getElementById("kickoff-auto");
    const manuel = document.getElementById("kickoff-manuel");
    const temps  = document.getElementById("kickoff-auto-time");
    if (!input) return;
    if (k && k.status === "AUTO_CONFIRMED" && k.kickoff_s !== null && k.kickoff_s !== undefined) {
        input.value = k.kickoff_s;                       // transmis au pipeline : pas de seconde détection
        if (temps)  temps.textContent = _mmss(k.kickoff_s);
        if (auto)   auto.style.display = "flex";
        if (manuel) manuel.style.display = "none";
    } else {
        input.value = "";                                // vide : le pipeline fera sa propre détection
        if (auto)   auto.style.display = "none";
        if (manuel) manuel.style.display = "block";
    }
}

function toggleKickoffManuel(afficher) {
    const auto   = document.getElementById("kickoff-auto");
    const manuel = document.getElementById("kickoff-manuel");
    const raison = document.getElementById("kickoff-manuel-raison");
    if (afficher) {
        if (auto)   auto.style.display = "none";
        if (manuel) manuel.style.display = "block";
        if (raison) raison.textContent = "Indiquez le coup d'envoi (minutes:secondes dans la vidéo), ou laissez vide pour une détection automatique pendant l'analyse.";
    }
}

function updateKickoffFromManuel() {
    const champ = document.getElementById("kickoff-manuel-mmss");
    const input = document.getElementById("kickoff-s-input");
    if (!champ || !input) return;
    const m = (champ.value || "").trim().match(/^(\d+):([0-5]\d)$/);
    input.value = m ? (parseInt(m[1], 10) * 60 + parseInt(m[2], 10)) : "";   // invalide/vide : détection automatique
}

function selectMyTeam(tid) {
    const champ = document.getElementById("user-team-id");
    if (champ) champ.value = String(tid);
    for (const t of [0, 1]) {
        const btn = document.getElementById(`my-team-btn-${t}`);
        if (!btn) continue;
        const choisi = (t === tid);
        btn.style.borderColor = choisi ? "var(--cyan)" : "var(--border)";
        btn.style.color       = choisi ? "var(--cyan)" : "var(--gray)";
        btn.textContent       = choisi ? "✓ Mon équipe" : "🏠 C'est mon équipe";
    }
}
