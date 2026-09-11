# set_admin.py

from app import app, db, User

with app.app_context():
    db.create_all()  # sécurité — crée les tables si manquantes

    user = User.query.filter_by(email="mikarocco@hotmail.com").first()

    if not user:
        print("Utilisateur non trouve — cree d'abord ton compte sur /register")
    else:
        user.is_admin = True
        db.session.commit()
        print(f"Admin OK -> {user.email}")

# V5.2 (11/09/2026) : route /make-admin RETIREE - etait exposee
# publiquement sans AUCUNE protection (donnait les droits admin au
# premier utilisateur de la base a quiconque visitait cette URL, en
# production comme en dev). Utiliser desormais UNIQUEMENT ce script
# en ligne de commande (python Set_Admin.py) pour se donner les
# droits admin - jamais expose via HTTP.