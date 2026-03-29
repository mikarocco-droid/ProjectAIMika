// static/app.js
// -*- coding: utf-8 -*-

// ─────────────────────────────────────────
// UPLOAD DRAG & DROP
// ─────────────────────────────────────────
const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileName  = document.getElementById("file-name");
const submitBtn = document.getElementById("submit-btn");
const browseBtn = document.getElementById("browse-btn");

if (dropZone) {
    if (browseBtn) browseBtn.addEventListener("click", () => fileInput.click());

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
        const file = e.dataTransfer.files[0];
        if (file) setFile(file);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) setFile(fileInput.files[0]);
    });

    function setFile(file) {
        fileName.textContent = `📎 ${file.name} (${(file.size / 1024 / 1024).toFixed(0)} MB)`;
        updateSubmitBtn();
    }
}

// ─────────────────────────────────────────
// CONFIG PAR SPORT
// ─────────────────────────────────────────
const SPORTS_CONFIG = {
    "football": {
        icon: "⚽", hasGardien: true, hasNumero: true,
        matchDesc: "Buts, tirs, passes clés, highlights tactiques",
        joueurDesc: "Analyse complète par poste",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien equipe domicile|🥅 Gardien dom.", "gardien equipe visiteur|🥅 Gardien vis."] },
            { group: "🛡️ Défense", opts: ["défenseur central|Défenseur central", "latéral droit|Latéral droit", "latéral gauche|Latéral gauche"] },
            { group: "⚙️ Milieu",  opts: ["milieu défensif|Milieu défensif", "milieu central|Milieu central", "milieu offensif|Milieu offensif", "ailier droit|Ailier droit", "ailier gauche|Ailier gauche"] },
            { group: "⚡ Attaque", opts: ["avant-centre|Avant-centre", "attaquant droit|Att. droit", "attaquant gauche|Att. gauche"] },
        ]
    },
    "mini-foot": {
        icon: "⚽", hasGardien: true, hasNumero: true,
        matchDesc: "Buts, dribbles, passes, highlights futsal",
        joueurDesc: "Analyse complète par poste futsal",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien equipe domicile|🥅 Gardien dom.", "gardien equipe visiteur|🥅 Gardien vis."] },
            { group: "👟 Joueurs", opts: ["pivot|Pivot", "ailier droit|Ailier droit", "ailier gauche|Ailier gauche", "fixo|Fixo (défenseur)"] },
        ]
    },
    "basketball": {
        icon: "🏀", hasGardien: false, hasNumero: true,
        matchDesc: "Paniers, passes décisives, rebonds, highlights",
        joueurDesc: "Analyse par poste basketball",
        positions: [
            { group: "🏀 Postes", opts: ["meneur|Meneur (PG)", "arrière|Arrière (SG)", "ailier|Ailier (SF)", "ailier fort|Ailier fort (PF)", "pivot|Pivot (C)"] },
        ]
    },
    "handball": {
        icon: "🤾", hasGardien: true, hasNumero: true,
        matchDesc: "Buts, tirs, passes, actions défensives",
        joueurDesc: "Analyse par poste handball",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien equipe domicile|🥅 Gardien dom.", "gardien equipe visiteur|🥅 Gardien vis."] },
            { group: "🤾 Joueurs", opts: ["ailier droit|Ailier droit", "ailier gauche|Ailier gauche", "arrière droit|Arrière droit", "arrière gauche|Arrière gauche", "demi-centre|Demi-centre", "pivot|Pivot"] },
        ]
    },
    "rugby": {
        icon: "🏉", hasGardien: false, hasNumero: true,
        matchDesc: "Essais, plaquages, mêlées, touches, highlights",
        joueurDesc: "Analyse par poste rugby",
        positions: [
            { group: "🏉 Avants",   opts: ["pilier droit|Pilier droit", "pilier gauche|Pilier gauche", "talonneur|Talonneur", "2ème ligne|2ème ligne", "flanker|Flanker", "numéro 8|Numéro 8"] },
            { group: "🏃 Arrières", opts: ["demi de mêlée|Demi de mêlée", "demi d'ouverture|Demi d'ouverture", "centre|Centre", "ailier|Ailier", "arrière|Arrière"] },
        ]
    },
    "hockey sur glace": {
        icon: "🏒", hasGardien: true, hasNumero: true,
        matchDesc: "Buts, tirs, mises en jeu, highlights",
        joueurDesc: "Analyse par poste hockey glace",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien equipe domicile|🥅 Gardien dom.", "gardien equipe visiteur|🥅 Gardien vis."] },
            { group: "🏒 Joueurs", opts: ["défenseur droit|Défenseur droit", "défenseur gauche|Défenseur gauche", "ailier droit|Ailier droit", "ailier gauche|Ailier gauche", "centre|Centre"] },
        ]
    },
    "hockey sur gazon": {
        icon: "🏑", hasGardien: true, hasNumero: true,
        matchDesc: "Buts, tirs, passes, corners, highlights",
        joueurDesc: "Analyse par poste hockey gazon",
        positions: [
            { group: "🥅 Gardien", opts: ["gardien equipe domicile|🥅 Gardien dom.", "gardien equipe visiteur|🥅 Gardien vis."] },
            { group: "🏑 Joueurs", opts: ["défenseur|Défenseur", "milieu|Milieu", "avant|Avant", "attaquant centre|Att. centre"] },
        ]
    },
    "tennis": {
        icon: "🎾", hasGardien: false, hasNumero: false,
        matchDesc: "Points clés, aces, fautes, highlights par set",
        joueurDesc: "Analyse d'un joueur : service, retour, déplacements",
        positions: [
            { group: "🎾 Joueur", opts: ["joueur côté droit|Joueur côté droit (ad)", "joueur côté gauche|Joueur côté gauche (deuce)"] },
        ]
    },
    "tennis de table": {
        icon: "🏓", hasGardien: false, hasNumero: false,
        matchDesc: "Points clés, smashes, services, highlights",
        joueurDesc: "Analyse d'un joueur : attaque, défense, service",
        positions: [
            { group: "🏓 Joueur", opts: ["joueur côté droit|Joueur côté droit", "joueur côté gauche|Joueur côté gauche"] },
        ]
    },
    "padel": {
        icon: "🎾", hasGardien: false, hasNumero: false,
        matchDesc: "Points clés, vitrages, bandeja, highlights",
        joueurDesc: "Analyse d'un joueur : positionnement, coups, déplacements",
        positions: [
            { group: "🎾 Position", opts: ["joueur côté droit|Côté droit", "joueur côté gauche|Côté gauche"] },
        ]
    },
};

// ─────────────────────────────────────────
// ÉLÉMENTS
// ─────────────────────────────────────────
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
// CONSTRUCTION POSITIONS
// ─────────────────────────────────────────
function buildPositions(sport) {
    const cfg = SPORTS_CONFIG[sport];
    if (!positionEl || !cfg) return;
    positionEl.innerHTML = '<option value="">-- Choisir --</option>';
    cfg.positions.forEach(g => {
        const grp = document.createElement("optgroup");
        grp.label = g.group;
        g.opts.forEach(o => {
            const [val, label] = o.split("|");
            const opt = document.createElement("option");
            opt.value       = val;
            opt.textContent = label;
            grp.appendChild(opt);
        });
        positionEl.appendChild(grp);
    });
}

// ─────────────────────────────────────────
// MISE À JOUR SPORT
// ─────────────────────────────────────────
function updateSport() {
    const sport = sportSelect ? sportSelect.value : "football";
    const cfg   = SPORTS_CONFIG[sport] || SPORTS_CONFIG["football"];

    if (matchIcon)  matchIcon.textContent  = cfg.icon;
    if (matchDesc)  matchDesc.textContent  = cfg.matchDesc;
    if (joueurDesc) joueurDesc.textContent = cfg.joueurDesc;

    if (fieldNumero)    fieldNumero.style.display    = cfg.hasNumero  ? "block" : "none";
    if (gardienSection) gardienSection.style.display = cfg.hasGardien ? "block" : "none";

    buildPositions(sport);
    updateSubmitBtn();
}

// ─────────────────────────────────────────
// TOGGLE MODE JOUEUR
// ─────────────────────────────────────────
function toggleJoueurOptions() {
    if (joueurOpts) {
        joueurOpts.style.display = modeJoueur && modeJoueur.checked ? "block" : "none";
    }
    updateSubmitBtn();
}

// ─────────────────────────────────────────
// MISE À JOUR BOUTON SUBMIT
// ─────────────────────────────────────────
function updateSubmitBtn() {
    if (!submitBtn) return;

    const sport    = sportSelect ? sportSelect.value : "football";
    const cfg      = SPORTS_CONFIG[sport] || SPORTS_CONFIG["football"];
    const hasFile  = fileInput && fileInput.files && fileInput.files.length > 0;
    const isJoueur = modeJoueur && modeJoueur.checked;

    // Pas de fichier
    if (!hasFile) {
        submitBtn.disabled    = true;
        submitBtn.textContent = "Selectionnez une video";
        submitBtn.style.opacity = "0.6";
        return;
    }

    // Mode joueur — vérifier les champs obligatoires
    if (isJoueur) {
        const hasNum     = !cfg.hasNumero || (numJoueur && numJoueur.value.trim() !== "");
        const hasCouleur = couleurEl && couleurEl.value !== "";
        const hasPos     = positionEl && positionEl.value !== "";

        if (!hasCouleur || !hasPos || !hasNum) {
            submitBtn.disabled    = true;
            submitBtn.textContent = "Remplissez les infos du joueur";
            submitBtn.style.opacity = "0.6";
            return;
        }

        const num = cfg.hasNumero && numJoueur && numJoueur.value
            ? `#${numJoueur.value} ` : "";
        submitBtn.disabled      = false;
        submitBtn.style.opacity = "1";
        submitBtn.textContent   = `Analyser ${num}${couleurEl.value} — ${positionEl.options[positionEl.selectedIndex].text} →`;
        return;
    }

    // Mode match — fichier suffit
    submitBtn.disabled      = false;
    submitBtn.style.opacity = "1";
    submitBtn.textContent   = `Analyser ${cfg.icon} ${sport} →`;
}

// ─────────────────────────────────────────
// EVENTS FORM
// ─────────────────────────────────────────
if (sportSelect) sportSelect.addEventListener("change", updateSport);
if (modeMatch)   modeMatch.addEventListener("change",   toggleJoueurOptions);
if (modeJoueur)  modeJoueur.addEventListener("change",  toggleJoueurOptions);
if (numJoueur)   numJoueur.addEventListener("input",    updateSubmitBtn);
if (couleurEl)   couleurEl.addEventListener("change",   updateSubmitBtn);
if (positionEl)  positionEl.addEventListener("change",  updateSubmitBtn);

// ─────────────────────────────────────────
// INDICATEUR UPLOAD EN COURS
// ─────────────────────────────────────────
const uploadForm = document.getElementById("upload-form");

if (uploadForm) {
    uploadForm.addEventListener("submit", () => {

        // Désactiver bouton
        if (submitBtn) {
            submitBtn.disabled      = true;
            submitBtn.textContent   = "Upload en cours...";
            submitBtn.style.opacity = "0.7";
        }

        // Zone de drop — feedback visuel
        if (dropZone) {
            dropZone.style.borderColor = "#2563eb";
            dropZone.style.background  = "#eff6ff";
        }

        // Message avec spinner
        if (fileName) {
            fileName.innerHTML = `
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    justify-content: center;
                    margin-top: 10px;
                ">
                    <div class="spinner"></div>
                    <span style="color:#2563eb;font-weight:500;">
                        Upload en cours — ne fermez pas cette page...
                    </span>
                </div>
            `;
        }
    });
}

// ─────────────────────────────────────────
// INIT AU CHARGEMENT
// ─────────────────────────────────────────
updateSport();

// ─────────────────────────────────────────
// POLLING PROGRESSION ANALYSES
// ─────────────────────────────────────────
document.querySelectorAll(".progress-container").forEach(el => {
    const id = el.dataset.id;
    if (id) poll(id, el);
});

function poll(id, el) {
    setTimeout(async () => {
        try {
            const r    = await fetch(`/api/status/${id}`);
            const data = await r.json();

            const fill = el.querySelector(".progress-fill");
            const msg  = el.querySelector(".progress-msg");

            if (fill) fill.style.width = (data.progress || 0) + "%";
            if (msg)  msg.textContent  = data.progress_msg || "En cours...";

            if (data.status === "done") {
                // Recharger pour afficher "Voir les résultats"
                location.reload();

            } else if (data.status === "error") {
                if (msg) {
                    msg.textContent = "Erreur : " + (data.progress_msg || "inconnue");
                    msg.style.color = "#dc2626";
                }
                // Pas de retry sur erreur

            } else {
                // Continuer à poller
                poll(id, el);
            }

        } catch {
            // Retry silencieux si réseau instable
            poll(id, el);
        }
    }, 3000);
}