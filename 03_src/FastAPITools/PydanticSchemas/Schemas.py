from typing import Literal,List
from pydantic import BaseModel, Field, ConfigDict


class ObservationsEntree(BaseModel):
    """Les 71 entrées du modèle ; historiques strictement passés, sans cible ni score dérivé."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=True)

    montant_eur: float = Field(gt=0, description="Montant du paiement en euros.")
    categorie_marchand: Literal['Alimentation', 'Boulangerie', 'Carburant', 'Habillement', 'Hébergement', 'Loisirs', 'Pharmacie', 'Restauration', 'Services numériques', 'Transport', 'Électronique', 'Équipement de la maison']
    pays_marchand: str = Field(pattern=r"^[A-Z]{2}$", description="Code pays du marchand sur deux lettres majuscules.")
    pays_ip: str | None = Field(default=None, pattern=r"^[A-Z]{2}$", description="Code pays IP, ou null si indisponible.")
    canal: Literal['Application', 'Magasin', 'Site web', 'À distance']
    mode_saisie: Literal['Carte enregistrée', 'Portefeuille mobile', 'Puce', 'Saisie en ligne', 'Sans contact']
    heure: int = Field(ge=0, le=23, description="Heure du paiement en Europe/Paris.")
    distance_domicile_marchand_km: float | None = Field(default=None, ge=0, le=20_016, description="Distance Haversine en km ; null si coordonnées manquantes.")
    delai_precedent_carte_h: float | None = Field(default=None, gt=0, description="Heures depuis le paiement strictement précédent ; null sans historique.")
    nb_carte_1h_passe: int = Field(ge=0, description="Paiements de la carte dans l’heure précédente, paiement courant exclu.")
    nb_carte_jour_passe: int = Field(ge=0, description="Paiements de la carte depuis minuit Paris, avant le paiement courant.")
    nb_appareils_carte_passe: int = Field(ge=0, description="Appareils distincts déjà observés pour la carte.")
    anciennete_carte_j: float = Field(ge=0, description="Jours depuis le premier paiement observé de la carte ; 0 au premier paiement.")
    ecart_mediane_enseigne_eur: float | None = Field(default=None, ge=0, description="Écart absolu à la médiane passée de l’enseigne, en EUR ; null sans passé.")
    ecart_mediane_carte_eur: float | None = Field(default=None, ge=0, description="Écart absolu à la médiane passée de la carte, en EUR ; null sans passé.")

    # Parts historiques par heure : 0 à 1 ; null sans historique client.
    part_client_heure_00_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_01_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_02_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_03_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_04_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_05_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_06_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_07_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_08_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_09_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_10_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_11_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_12_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_13_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_14_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_15_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_16_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_17_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_18_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_19_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_20_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_21_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_22_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_heure_23_passe: float | None = Field(default=None, ge=0, le=1)

    # Parts historiques par categorie_marchand : 0 à 1 ; null sans historique client.
    part_client_categorie_marchand_alimentation_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_boulangerie_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_carburant_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_habillement_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_hebergement_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_loisirs_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_pharmacie_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_restauration_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_services_numeriques_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_transport_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_electronique_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_categorie_marchand_equipement_de_la_maison_passe: float | None = Field(default=None, ge=0, le=1)

    # Parts historiques par pays_marchand : 0 à 1 ; null sans historique client.
    part_client_pays_marchand_be_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_ca_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_ch_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_de_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_es_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_fr_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_gb_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_it_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_nl_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_pt_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_pays_marchand_us_passe: float | None = Field(default=None, ge=0, le=1)

    # Parts historiques par canal : 0 à 1 ; null sans historique client.
    part_client_canal_application_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_canal_magasin_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_canal_site_web_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_canal_a_distance_passe: float | None = Field(default=None, ge=0, le=1)

    # Parts historiques par mode_saisie : 0 à 1 ; null sans historique client.
    part_client_mode_saisie_carte_enregistree_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_mode_saisie_portefeuille_mobile_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_mode_saisie_puce_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_mode_saisie_saisie_en_ligne_passe: float | None = Field(default=None, ge=0, le=1)
    part_client_mode_saisie_sans_contact_passe: float | None = Field(default=None, ge=0, le=1)


class BatchEntree(BaseModel):

    data: List[ObservationsEntree] = Field(min_length=1, max_length=10)


class ObservationsSortie(BaseModel):

    id_observation: int
    model_proba: float = Field(gt=0, lt=100)


class BatchSortie(BaseModel):

    model_version: str
    model_nb_probas: int
    model_probas: List[ObservationsSortie]


class ShapValue(BaseModel):

    name: str
    value: float | None
    shap_value: float


class ShapInfo(BaseModel):

    id_observation: int
    base_value: float
    prediction: float = Field(ge=0, le=100)
    features: List[ShapValue] = Field(min_length=11, max_length=11)


class ShapSortie(BaseModel):

    model_version: str
    output_scale: Literal["percentage_points"] = "percentage_points"
    explanations: List[ShapInfo] = Field(min_length=1, max_length=10)
