# payments/stripe_handler.py
# -*- coding: utf-8 -*-

import config

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    print("⚠️  stripe non installé — paiement désactivé")


# FIX — init uniquement si clé présente
if STRIPE_AVAILABLE and config.STRIPE_SECRET_KEY:
    stripe.api_key = config.STRIPE_SECRET_KEY
elif STRIPE_AVAILABLE:
    print("⚠️  STRIPE_SECRET_KEY manquante — paiement désactivé")


# ─────────────────────────────────────────
# CHECKOUT SESSION
# ─────────────────────────────────────────
def create_checkout_session(user_email, plan, success_url, cancel_url):
    if not STRIPE_AVAILABLE or not config.STRIPE_SECRET_KEY:
        return None

    price_id = config.STRIPE_PRICE_IDS.get(plan)
    if not price_id:
        print(f"❌ Plan inconnu ou price_id manquant : {plan}")
        return None

    # Paiement unique (plan "unique") → mode one-time
    mode = "payment" if plan == "unique" else "subscription"

    try:
        session = stripe.checkout.Session.create(
            payment_method_types = ["card"],
            mode                 = mode,
            customer_email       = user_email,
            line_items           = [{"price": price_id, "quantity": 1}],
            metadata             = {"plan": plan, "email": user_email},
            success_url          = success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url           = cancel_url
        )
        return session.url

    except stripe.error.StripeError as e:
        print(f"❌ Stripe checkout error : {e}")
        return None

    except Exception as e:
        print(f"❌ Erreur inattendue checkout : {e}")
        return None


# ─────────────────────────────────────────
# PORTAIL CLIENT
# ─────────────────────────────────────────
def create_portal_session(stripe_customer_id, return_url):
    if not STRIPE_AVAILABLE or not config.STRIPE_SECRET_KEY:
        return None

    try:
        session = stripe.billing_portal.Session.create(
            customer   = stripe_customer_id,
            return_url = return_url
        )
        return session.url

    except stripe.error.StripeError as e:
        print(f"❌ Stripe portal error : {e}")
        return None

    except Exception as e:
        print(f"❌ Erreur inattendue portal : {e}")
        return None


# ─────────────────────────────────────────
# VÉRIFICATION SESSION APRÈS PAIEMENT
# ─────────────────────────────────────────
def verify_checkout_session(session_id):
    if not STRIPE_AVAILABLE or not config.STRIPE_SECRET_KEY:
        return None

    try:
        session = stripe.checkout.Session.retrieve(session_id)

        if session.payment_status != "paid":
            print(f"⚠️  Session non payée : {session.payment_status}")
            return None

        return {
            "plan":            session.metadata.get("plan"),
            "email":           session.metadata.get("email"),
            "customer_id":     session.customer,
            "subscription_id": session.subscription
        }

    except stripe.error.StripeError as e:
        print(f"❌ Stripe verify error : {e}")
        return None

    except Exception as e:
        print(f"❌ Erreur inattendue verify : {e}")
        return None


# ─────────────────────────────────────────
# WEBHOOK
# ─────────────────────────────────────────
def handle_webhook(payload, sig_header):
    if not STRIPE_AVAILABLE or not config.STRIPE_WEBHOOK_SECRET:
        return None

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, config.STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        print("❌ Webhook signature invalide")
        return None
    except Exception as e:
        print(f"❌ Webhook error : {e}")
        return None

    event_type = event["type"]
    data       = event["data"]["object"]

    # ── Paiement initial réussi ───────────
    if event_type == "checkout.session.completed":
        return {
            "event_type":      "payment_success",
            "plan":            data.get("metadata", {}).get("plan"),
            "email":           data.get("metadata", {}).get("email"),
            "customer_id":     data.get("customer"),
            "subscription_id": data.get("subscription")
        }

    # ── Renouvellement abonnement ─────────
    elif event_type == "invoice.payment_succeeded":
        sub = data.get("subscription")
        if sub:
            try:
                subscription = stripe.Subscription.retrieve(sub)
                plan = subscription.metadata.get("plan", "pro")
            except Exception:
                plan = "pro"

            return {
                "event_type":      "renewal_success",
                "plan":            plan,
                "customer_id":     data.get("customer"),
                "subscription_id": sub
            }

    # ── Paiement échoué ───────────────────
    elif event_type == "invoice.payment_failed":
        return {
            "event_type":  "payment_failed",
            "customer_id": data.get("customer")
        }

    # ── Résiliation abonnement ────────────
    elif event_type in [
        "customer.subscription.deleted",
        "customer.subscription.canceled"
    ]:
        return {
            "event_type":  "subscription_canceled",
            "customer_id": data.get("customer")
        }

    # Événement non géré — on retourne None silencieusement
    return None