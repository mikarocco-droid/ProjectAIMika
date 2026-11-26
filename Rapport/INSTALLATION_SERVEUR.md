# ScoutIA — installation en production (26/11/2026)

Architecture : **site sur un petit serveur Hetzner** (Flask + PostgreSQL + Caddy pour le HTTPS), **vidéos et résultats
en transit sur Cloudflare R2**, **analyses sur des machines GPU RunPod créées à la demande** (une par analyse rapide, tout de suite ;
limite de sécurité `SCOUTIA_GPU_MAX`), supprimées après `SCOUTIA_GPU_INACTIF_MIN` minutes sans travail. Pas de Redis : la file d'attente
est dans PostgreSQL. Les machines GPU n'ont **pas** la clé RunPod : c'est le site qui les crée et les supprime.

## 0. Avant tout (sécurité)
- Supprimer `CleAPIStripe.txt` de GitHub **et régénérer les clés Stripe** : un fichier supprimé reste dans
  l'historique Git.
- Les secrets ne vont que dans `param.env` sur le serveur (jamais dans le dépôt, jamais dans une conversation).

## 1. Comptes (au nom de l'activité, une fois le numéro d'entreprise obtenu)
1. **Domaine** (ex. `.be` chez OVH ou Infomaniak). Zone DNS : `A @ → IP du serveur`, `A www → IP du serveur`.
2. **Hetzner Cloud** : serveur CX23 (2 vCPU, 4 Go ; 8 Go si besoin plus tard), Ubuntu 24.04, Allemagne, votre clé SSH,
   pare-feu 22 / 80 / 443. Plus tard, si les résultats prennent de la place : un **Volume** Hetzner monté sur
   `/opt/scoutia/donnees`.
3. **Cloudflare R2** : bucket privé (ex. `scoutia`). Deux jetons « lecture et écriture » limités à ce bucket :
   un pour le **site**, un pour les **machines GPU** (révocable séparément). Déposer
   `parseq_scoutia_maillots_v2.pt` dans le bucket sous `modeles/parseq_scoutia_maillots_v2.pt`.
4. **RunPod** : crédit prépayé (10 à 20 $), une clé API → **uniquement** dans le `param.env` du serveur.
5. **GitHub Container Registry (GHCR)** : rien à créer, c'est votre compte GitHub (jeton « write:packages »).

## 2. Installer le serveur (en SSH : `ssh root@IP`)
```
apt update && apt -y upgrade
curl -fsSL https://get.docker.com | sh
mkdir -p /opt/scoutia && cd /opt/scoutia
git clone https://github.com/mikarocco-droid/ProjectAIMika.git .
nano param.env                 # voir la liste ci-dessous
nano Caddyfile                 # remplacer matchscoutia.com par votre domaine
echo "POSTGRES_PASSWORD=$(openssl rand -hex 24)" > .env
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml logs -f app      # vérifier le démarrage
```
Le site répond en HTTPS sur votre domaine dès que le DNS est pris en compte (certificat automatique).

### `param.env` (serveur)
```
SECRET_KEY=<64 caractères aléatoires : openssl rand -hex 32>
SCOUTIA_SITE_URL=https://votre-domaine
SCOUTIA_WORKER_TOKEN=<autre secret : openssl rand -hex 32>
GEMINI_API_KEY=...
STRIPE_SECRET_KEY=...            STRIPE_PUBLIC_KEY=...        STRIPE_WEBHOOK_SECRET=...
R2_ACCOUNT_ID=...   R2_ACCESS_KEY_ID=...   R2_SECRET_ACCESS_KEY=...   R2_BUCKET_NAME=scoutia
RUNPOD_API_KEY=...
RUNPOD_IMAGE=ghcr.io/<votre-compte-github>/scoutia-gpu:2026-11-26
RUNPOD_GPU_TYPES=NVIDIA L4,NVIDIA RTX A5000,NVIDIA GeForce RTX 4090
RUNPOD_CLOUD_TYPE=SECURE
SCOUTIA_GPU_MAX=10              # limite de SÉCURITÉ : chaque analyse rapide a sa machine tout de suite
SCOUTIA_GPU_MAX_48H=1           # machines créées pour les analyses « sous 48 h » (elles peuvent attendre)
SCOUTIA_GPU_INACTIF_MIN=15
SCOUTIA_PROFIL_KO=2026-11
SCOUTIA_JOUEUR_V30=1
SCOUTIA_CAMERA_TYPE=high_side
SCOUTIA_RESUME_MAX_MIN=12
SCOUTIA_RESUME_COPIE=1
RETENTION_JOURS=30               # résultats gardés 30 jours pour tous (les jetons remplacent les formules)
ADMIN_EMAIL=votre-adresse        # erreurs signalées par les utilisateurs → e-mail avec le lien du paquet de rejeu
SMTP_HOST=ssl0.ovh.net           # (exemple OVH : à vérifier selon votre boîte mail)
SMTP_PORT=587
SMTP_USER=contact@votre-domaine
SMTP_PASS=...
```
Jeton R2 des machines GPU (le 2e jeton de l'étape 1.3) : ajouter `R2_ACCESS_KEY_ID_GPU=...` et
`R2_SECRET_ACCESS_KEY_GPU=...`. Sans eux, les machines GPU reçoivent le jeton du site.

## 3. Publier l'image de la machine GPU (une fois, puis à chaque mise à jour du code d'analyse)
Sur le serveur (ou sur un PC avec Docker ; l'image fait ~10 à 15 Go, prévoir ~30 Go libres) :
```
cd /opt/scoutia && git pull
docker build -f Dockerfile.runpod -t ghcr.io/<compte>/scoutia-gpu:2026-11-26 .
echo <jeton GitHub write:packages> | docker login ghcr.io -u <compte> --password-stdin
docker push ghcr.io/<compte>/scoutia-gpu:2026-11-26
docker image prune -af          # libérer la place sur le serveur
```
Sur GitHub (Packages → scoutia-gpu → Package settings) : visibilité **publique** (RunPod la télécharge sans mot de
passe ; elle ne contient aucun secret grâce à `.dockerignore`), ou privée avec des identifiants enregistrés dans RunPod.

## 4. Premier test (petite vidéo d'abord)
1. Créer un compte sur le site, lui ajouter des jetons (page Administration → « Ajouter »).
2. Envoyer une vidéo courte (5 à 10 min) en « Rapide ».
3. Journal du site : `docker compose -f docker-compose.prod.yml logs -f app` → « [GPU] machine créée ».
4. Console RunPod : la machine apparaît ; son journal affiche « Machine GPU ScoutIA prête », puis « ANALYSE #… ».
5. À la fin : la page de résultats s'ouvre sur le site ; 15 min plus tard, la machine disparaît de RunPod.
6. Puis un match complet (SRU) : **mesurer la durée, le coût GPU (console RunPod) et le coût Gemini** → ajuster les
   prix des jetons.

## 5. Stripe (mode test, puis réel)
Webhook vers `https://votre-domaine/webhook/stripe`, événement `checkout.session.completed`. Activer Bancontact.

## Mises à jour
```
cd /opt/scoutia && git pull && docker compose -f docker-compose.prod.yml up -d --build
```
(et republier l'image GPU si le code d'analyse a changé, avec une nouvelle étiquette dans `RUNPOD_IMAGE`).

## Sauvegardes
`/opt/scoutia/donnees` (base PostgreSQL, résultats). Activer les sauvegardes Hetzner (≈ 20 % du prix du serveur) ou
un `pg_dump` quotidien vers R2.

## Erreurs signalées par les utilisateurs
Quand un utilisateur écarte un but que l'analyse avait **confirmé**, ajoute un **but oublié** ou corrige l'**équipe**
d'un but, un e-mail part vers `ADMIN_EMAIL` (au plus toutes les 15 min) avec le détail de tous les buts, les bornes,
les tenues et le lien **« paquet de rejeu »** (aussi dans Administration → bouton « ⬇ Rejeu »). À télécharger **avant
la date indiquée** (suppression automatique après `RETENTION_JOURS`). Le paquet contient le cache du suivi
(`cache_<id>_reference/`), les réponses Gemini (`gemini_cache_<id>.jsonl`), `verite_utilisateur.json` et le rapport :
le déposer dans un dataset Kaggle pour rejouer le match à 0 $.
