// static/app.js — ScoutIA V9.6

// ─────────────────────────────────────────
// ELEMENTS
// ─────────────────────────────────────────
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const fileName   = document.getElementById("file-name");
const submitBtn  = document.getElementById("submit-btn");
const browseBtn  = document.getElementById("browse-btn");

const sportSelect    = document.getElementById("sport-select");
const modeMatch      = document.getElementById("mode-match");
const modeJoueur     = document.getElementById("mode-joueur");

const joueurOpts     = document.getElementById("joueur-options");
const numJoueur      = document.getElementById("numero-joueur");
const couleurEl      = document.getElementById("couleur-maillot");
const positionEl     = document.getElementById("position-joueur");

const gardienSection = document.getElementById("gardien-section");
const fieldNumero    = document.getElementById("field-numero");
const matchIcon      = document.getElementById("mode-match-icon");
const matchDesc      = document.getElementById("mode-match-desc");
const joueurDesc     = document.getElementById("mode-joueur-desc");


// ─────────────────────────────────────────
// CONFIG SPORTS
// ─────────────────────────────────────────
const SPORTS_CONFIG = {
    football: {
        icon: "⚽",
        hasGardien: true,
        hasNumero: true,
        matchDesc: "Highlights, rapport tactique, xG, top joueurs",
        joueurDesc: "Rapport complet sur un joueur spécifique",
        positions: [
            { group: "🥅 Gardien",  opts: ["gardien|Gardien"] },
            { group: "🛡️ Défense",  opts: ["defenseur|Défenseur", "lateraldroit|Latéral droit", "lateralgauche|Latéral gauche"] },
            { group: "⚙️ Milieu",   opts: ["milieu|Milieu", "milieud|Milieu défensif", "milieuof|Milieu offensif", "ailier|Ailier"] },
            { group: "⚡ Attaque",  opts: ["attaquant|Attaquant", "avant-centre|Avant-centre"] },
        ]
    },
    "mini-foot": {
        icon: "⚽",
        hasGardien: true,
        hasNumero: true,
        matchDesc: "Analyse mini-foot / futsal",
        joueurDesc: "Rapport joueur futsal",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien|Gardien"] },
            { group: "⚡ Joueurs", opts: ["pivot|Pivot", "ala|Ala", "fixo|Fixo"] },
        ]
    },
    basketball: {
        icon: "🏀",
        hasGardien: false,
        hasNumero: true,
        matchDesc: "Stats basket, highlights, rapport",
        joueurDesc: "Rapport poste basket",
        positions: [
            { group: "🏀 Postes", opts: ["pg|Meneur (PG)", "sg|Arrière (SG)", "sf|Ailier (SF)", "pf|Ailier fort (PF)", "c|Pivot (C)"] }
        ]
    },
    handball: {
        icon: "🤾",
        hasGardien: true,
        hasNumero: true,
        matchDesc: "Analyse handball complète",
        joueurDesc: "Rapport joueur handball",
        positions: [
            { group: "🥅 Gardien",  opts: ["gardien|Gardien"] },
            { group: "⚡ Joueurs",  opts: ["pivot|Pivot", "ailier|Ailier", "arrieredroit|Arrière droit", "arrierecentre|Demi-centre", "arrierecentreg|Arrière gauche"] },
        ]
    },
    tennis: {
        icon: "🎾",
        hasGardien: false,
        hasNumero: false,
        matchDesc: "Analyse tennis — stats, highlights",
        joueurDesc: "Rapport joueur tennis",
        positions: []
    },
};


// ─────────────────────────────────────────
// FILE HANDLING
// ─────────────────────────────────────────
function setFile(file) {
    if (!file) return;
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    if (fileName) {
        fileName.textContent = `${file.name}  (${sizeMB} MB)`;
        fileName.style.color = "var(--cyan)";
    }
    // Feedback visuel sur la drop zone
    if (dropZone) {
        dropZone.style.borderColor = "var(--cyan)";
        dropZone.style.background  = "var(--cyan-dim, rgba(0,200,230,0.06))";
    }
    updateSubmitBtn();
}

if (dropZone) {
    browseBtn?.addEventListener("click", () => fileInput?.click());

    dropZone.addEventListener("click", () => fileInput?.click());

    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file && fileInput) {
            // Injecter le fichier dans l'input
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
        }
        setFile(file);
    });

    fileInput?.addEventListener("change", () => {
        setFile(fileInput.files[0]);
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
        const optGroup = document.createElement("optgroup");
        optGroup.label = group.group;
        group.opts.forEach(o => {
            const [val, label] = o.split("|");
            const opt = document.createElement("option");
            opt.value = val;
            opt.textContent = label;
            optGroup.appendChild(opt);
        });
        positionEl.appendChild(optGroup);
    });
}


// ─────────────────────────────────────────
// UPDATE SPORT UI
// ─────────────────────────────────────────
function updateSport() {
    const sport = sportSelect?.value || "football";
    const cfg   = SPORTS_CONFIG[sport] || SPORTS_CONFIG.football;

    if (matchIcon) matchIcon.textContent = cfg.icon;
    if (matchDesc) matchDesc.textContent = cfg.matchDesc;
    if (joueurDesc) joueurDesc.textContent = cfg.joueurDesc;

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
    if (joueurOpts) {
        joueurOpts.style.display = isJoueur ? "block" : "none";
    }
    updateSubmitBtn();
}

modeMatch?.addEventListener("change", toggleJoueurOptions);
modeJoueur?.addEventListener("change", toggleJoueurOptions);
sportSelect?.addEventListener("change", updateSport);

numJoueur?.addEventListener("input",  updateSubmitBtn);
couleurEl?.addEventListener("change", updateSubmitBtn);
positionEl?.addEventListener("change", updateSubmitBtn);


// ─────────────────────────────────────────
// VALIDATION SUBMIT
// ─────────────────────────────────────────
function updateSubmitBtn() {
    if (!submitBtn) return;

    const sport    = sportSelect?.value || "football";
    const cfg      = SPORTS_CONFIG[sport] || SPORTS_CONFIG.football;
    const hasFile  = fileInput?.files?.length > 0;
    const isJoueur = modeJoueur?.checked;

    if (!hasFile) {
        submitBtn.disabled     = true;
        submitBtn.textContent  = "Sélectionnez une vidéo";
        return;
    }

    if (isJoueur) {
        const hasNum = !cfg.hasNumero || numJoueur?.value;
        const hasPos = positionEl?.value;
        const hasCol = couleurEl?.value;

        if (!hasNum || !hasPos || !hasCol) {
            submitBtn.disabled    = true;
            submitBtn.textContent = "Complétez les infos joueur";
            return;
        }

        submitBtn.disabled    = false;
        submitBtn.textContent = `Analyser le joueur ${cfg.icon}`;
        return;
    }

    submitBtn.disabled    = false;
    submitBtn.textContent = `Lancer l'analyse ${cfg.icon}`;
}


// ─────────────────────────────────────────
// FORM SUBMIT UX
// ─────────────────────────────────────────
const uploadForm = document.getElementById("upload-form");

if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
        if (submitBtn) {
            submitBtn.disabled    = true;
            submitBtn.textContent = "Envoi en cours...";
            submitBtn.style.opacity = "0.6";
        }
        if (fileName) {
            fileName.innerHTML = `<span style="color:var(--cyan)">Transfert vers le serveur...</span>`;
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

        const pct = Math.max(0, Math.min(100, data.progress || 0));

        if (fill) {
            fill.style.width      = pct + "%";
            fill.style.transition = "width 0.5s ease";
        }

        if (msg) {
            if (data.status === "error") {
                msg.textContent  = "Erreur : " + (data.progress_msg || "inconnue");
                msg.style.color  = "var(--red)";
            } else {
                msg.textContent = data.progress_msg || `Analyse en cours... ${pct}%`;
                msg.style.color = "";
            }
        }

        if (data.status === "done") {
            // Feedback visuel avant reload
            if (fill) {
                fill.style.width      = "100%";
                fill.style.background = "var(--green)";
            }
            if (msg) {
                msg.textContent = "Terminé !";
                msg.style.color = "var(--green)";
            }
            setTimeout(() => location.reload(), 800);

        } else if (data.status !== "error") {
            poll(id, el);
        }

    } catch (err) {
        // Retry silencieux sur erreur réseau
        console.warn(`Poll ${id} erreur :`, err.message);
        poll(id, el);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────
updateSport();
toggleJoueurOptions();
