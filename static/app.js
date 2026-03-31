// static/app.js (V15 FULL PRO)

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
// CONFIG SPORTS (IDENTIQUE MAIS CENTRALISE)
// ─────────────────────────────────────────
const SPORTS_CONFIG = {
    football: {
        icon: "⚽",
        hasGardien: true,
        hasNumero: true,
        matchDesc: "Analyse complète du match",
        joueurDesc: "Analyse individuelle joueur",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien|Gardien"] },
            { group: "⚙️ Milieu", opts: ["milieu|Milieu", "ailier|Ailier"] },
            { group: "⚡ Attaque", opts: ["attaquant|Attaquant"] },
        ]
    },
    basketball: {
        icon: "🏀",
        hasGardien: false,
        hasNumero: true,
        matchDesc: "Analyse basketball",
        joueurDesc: "Analyse poste basket",
        positions: [
            { group: "🏀 Postes", opts: ["pg|Meneur", "sg|Arrière", "c|Pivot"] }
        ]
    }
};


// ─────────────────────────────────────────
// FILE HANDLING
// ─────────────────────────────────────────
function setFile(file) {
    if (!file) return;

    const sizeMB = (file.size / 1024 / 1024).toFixed(1);

    fileName.textContent = `📎 ${file.name} (${sizeMB} MB)`;
    updateSubmitBtn();
}

if (dropZone) {
    browseBtn?.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.style.borderColor = "#2563eb";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "";
    });

    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.style.borderColor = "";
        setFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener("change", () => {
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
    const cfg = SPORTS_CONFIG[sport];

    if (!cfg) return;

    matchIcon.textContent  = cfg.icon;
    matchDesc.textContent  = cfg.matchDesc;
    joueurDesc.textContent = cfg.joueurDesc;

    fieldNumero.style.display    = cfg.hasNumero  ? "block" : "none";
    gardienSection.style.display = cfg.hasGardien ? "block" : "block";

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


// ─────────────────────────────────────────
// VALIDATION SUBMIT
// ─────────────────────────────────────────
function updateSubmitBtn() {
    if (!submitBtn) return;

    const sport = sportSelect?.value || "football";
    const cfg   = SPORTS_CONFIG[sport];

    const hasFile = fileInput?.files?.length > 0;
    const isJoueur = modeJoueur?.checked;

    if (!hasFile) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Sélectionnez une vidéo";
        return;
    }

    if (isJoueur) {
        const hasNum  = !cfg.hasNumero || numJoueur?.value;
        const hasPos  = positionEl?.value;
        const hasCol  = couleurEl?.value;

        if (!hasNum || !hasPos || !hasCol) {
            submitBtn.disabled = true;
            submitBtn.textContent = "Complétez les infos joueur";
            return;
        }

        submitBtn.disabled = false;
        submitBtn.textContent = `Analyser joueur →`;
        return;
    }

    submitBtn.disabled = false;
    submitBtn.textContent = `Analyser ${cfg.icon} ${sport}`;
}


// ─────────────────────────────────────────
// FORM SUBMIT UX
// ─────────────────────────────────────────
const uploadForm = document.getElementById("upload-form");

if (uploadForm) {
    uploadForm.addEventListener("submit", () => {

        submitBtn.disabled = true;
        submitBtn.textContent = "Analyse en cours...";
        submitBtn.style.opacity = "0.7";

        fileName.innerHTML = `
        <div style="display:flex;align-items:center;gap:10px;">
            <div class="spinner"></div>
            <span>Traitement IA en cours...</span>
        </div>`;
    });
}


// ─────────────────────────────────────────
// POLLING PROGRESSION
// ─────────────────────────────────────────
document.querySelectorAll(".progress-container").forEach(el => {
    const id = el.dataset.id;
    if (id) poll(id, el);
});

function poll(id, el) {
    setTimeout(async () => {
        try {
            const res = await fetch(`/api/status/${id}`);
            const data = await res.json();

            const fill = el.querySelector(".progress-fill");
            const msg  = el.querySelector(".progress-msg");

            if (fill) fill.style.width = (data.progress || 0) + "%";
            if (msg)  msg.textContent  = data.progress_msg || "Analyse en cours...";

            if (data.status === "done") {
                location.reload();
            } else if (data.status === "error") {
                if (msg) {
                    msg.textContent = "Erreur : " + data.progress_msg;
                    msg.style.color = "#dc2626";
                }
            } else {
                poll(id, el);
            }

        } catch {
            poll(id, el);
        }
    }, 3000);
}


// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────
updateSport();
toggleJoueurOptions();