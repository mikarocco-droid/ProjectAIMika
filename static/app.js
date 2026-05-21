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
        matchDesc:  "Highlights, rapport tactique, xG, top joueurs",
        joueurDesc: "Rapport complet sur un joueur spécifique",
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

function startUploadWithProgress(file) {
    if (!file) return;

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
    };

    xhr.onload = () => {
        if (progressWrap) progressWrap.style.display = "none";

        if (xhr.status === 200) {
            try {
                const data = JSON.parse(xhr.responseText);
                const uid  = data.upload_id || "";

                // Stocker l'upload_id dans le form
                if (previewUploadId) previewUploadId.value = uid;

                // Afficher "Analyse en cours" pendant la détection
                const stepDetecting = document.getElementById("step-detecting");
                if (stepDetecting) {
                    stepDetecting.style.display = "block";
                    stepDetecting.scrollIntoView({ behavior: "smooth", block: "start" });
                }

                function revealConfig() {
                    // Masquer detecting, révéler config
                    if (stepDetecting) stepDetecting.style.display = "none";
                    _uploadDone = true;
                    if (stepConfig) {
                        stepConfig.style.display = "block";
                        setTimeout(() => {
                            stepConfig.scrollIntoView({ behavior: "smooth", block: "start" });
                        }, 150);
                    }
                    updateSubmitBtn();
                }

                if (uid) {
                    runTeamDetection(uid).then(revealConfig).catch(revealConfig);
                } else {
                    revealConfig();
                }

            } catch (e) {
                showUploadError("Réponse inattendue du serveur");
            }
        } else {
            showUploadError("Erreur " + xhr.status);
        }
    };

    xhr.onerror = () => showUploadError("Erreur réseau");

    xhr.open("POST", "/api/upload-preview");
    xhr.send(fd);
}

function showUploadError(msg) {
    if (progressWrap) {
        progressWrap.innerHTML =
            `<div style="color:var(--red);font-size:0.82rem">❌ ${msg} — réessayez</div>`;
    }
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
            if (prog) prog.textContent = "Détection non disponible — renseignez manuellement.";
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
        }

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
        submitBtn.textContent = "Sélectionnez une vidéo";
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
