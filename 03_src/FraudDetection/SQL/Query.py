def query():
    """Renvoie la requête DuckDB qui construit les variables métier et historiques.

    La requête attend une relation ``df_copy`` contenant les paiements nettoyés.
    Elle calcule notamment les habitudes du client, la vélocité de la carte,
    les écarts aux médianes et la distance domicile–marchand.
    Les fenêtres historiques excluent le paiement courant et les autres
    paiements au même instant. Le calendrier local utilise Europe/Paris.
    Cette fonction retourne uniquement le SQL, sans l'exécuter.
    """

    requete = """
    WITH base AS (
        SELECT *,
            row_number() OVER () AS _ligne,
            CAST(date_paiement_utc AS TIMESTAMPTZ) AS ts_utc,
            timezone('Europe/Paris', CAST(date_paiement_utc AS TIMESTAMPTZ)) AS ts_paris
        FROM df_copy
    ), calendrier AS (
        SELECT *, CAST(ts_paris AS DATE) AS date_paris
        FROM base
    ), historiques AS (
        SELECT *,
            COUNT(*) OVER passe_client AS n_client_total,
            -- Achats passés du client par categorie_marchand.
            COUNT(*) FILTER (WHERE categorie_marchand = 'Alimentation') OVER passe_client AS n_client_categorie_marchand_alimentation,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Boulangerie') OVER passe_client AS n_client_categorie_marchand_boulangerie,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Carburant') OVER passe_client AS n_client_categorie_marchand_carburant,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Habillement') OVER passe_client AS n_client_categorie_marchand_habillement,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Hébergement') OVER passe_client AS n_client_categorie_marchand_hebergement,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Loisirs') OVER passe_client AS n_client_categorie_marchand_loisirs,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Pharmacie') OVER passe_client AS n_client_categorie_marchand_pharmacie,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Restauration') OVER passe_client AS n_client_categorie_marchand_restauration,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Services numériques') OVER passe_client AS n_client_categorie_marchand_services_numeriques,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Transport') OVER passe_client AS n_client_categorie_marchand_transport,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Électronique') OVER passe_client AS n_client_categorie_marchand_electronique,
            COUNT(*) FILTER (WHERE categorie_marchand = 'Équipement de la maison') OVER passe_client AS n_client_categorie_marchand_equipement_de_la_maison,
            -- Achats passés du client par pays_marchand.
            COUNT(*) FILTER (WHERE pays_marchand = 'BE') OVER passe_client AS n_client_pays_marchand_be,
            COUNT(*) FILTER (WHERE pays_marchand = 'CA') OVER passe_client AS n_client_pays_marchand_ca,
            COUNT(*) FILTER (WHERE pays_marchand = 'CH') OVER passe_client AS n_client_pays_marchand_ch,
            COUNT(*) FILTER (WHERE pays_marchand = 'DE') OVER passe_client AS n_client_pays_marchand_de,
            COUNT(*) FILTER (WHERE pays_marchand = 'ES') OVER passe_client AS n_client_pays_marchand_es,
            COUNT(*) FILTER (WHERE pays_marchand = 'FR') OVER passe_client AS n_client_pays_marchand_fr,
            COUNT(*) FILTER (WHERE pays_marchand = 'GB') OVER passe_client AS n_client_pays_marchand_gb,
            COUNT(*) FILTER (WHERE pays_marchand = 'IT') OVER passe_client AS n_client_pays_marchand_it,
            COUNT(*) FILTER (WHERE pays_marchand = 'NL') OVER passe_client AS n_client_pays_marchand_nl,
            COUNT(*) FILTER (WHERE pays_marchand = 'PT') OVER passe_client AS n_client_pays_marchand_pt,
            COUNT(*) FILTER (WHERE pays_marchand = 'US') OVER passe_client AS n_client_pays_marchand_us,
            -- Achats passés du client par canal.
            COUNT(*) FILTER (WHERE canal = 'Application') OVER passe_client AS n_client_canal_application,
            COUNT(*) FILTER (WHERE canal = 'Magasin') OVER passe_client AS n_client_canal_magasin,
            COUNT(*) FILTER (WHERE canal = 'Site web') OVER passe_client AS n_client_canal_site_web,
            COUNT(*) FILTER (WHERE canal = 'À distance') OVER passe_client AS n_client_canal_a_distance,
            -- Achats passés du client par mode_saisie.
            COUNT(*) FILTER (WHERE mode_saisie = 'Carte enregistrée') OVER passe_client AS n_client_mode_saisie_carte_enregistree,
            COUNT(*) FILTER (WHERE mode_saisie = 'Portefeuille mobile') OVER passe_client AS n_client_mode_saisie_portefeuille_mobile,
            COUNT(*) FILTER (WHERE mode_saisie = 'Puce') OVER passe_client AS n_client_mode_saisie_puce,
            COUNT(*) FILTER (WHERE mode_saisie = 'Saisie en ligne') OVER passe_client AS n_client_mode_saisie_saisie_en_ligne,
            COUNT(*) FILTER (WHERE mode_saisie = 'Sans contact') OVER passe_client AS n_client_mode_saisie_sans_contact,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 0) OVER passe_client AS n_client_heure_00,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 1) OVER passe_client AS n_client_heure_01,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 2) OVER passe_client AS n_client_heure_02,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 3) OVER passe_client AS n_client_heure_03,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 4) OVER passe_client AS n_client_heure_04,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 5) OVER passe_client AS n_client_heure_05,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 6) OVER passe_client AS n_client_heure_06,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 7) OVER passe_client AS n_client_heure_07,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 8) OVER passe_client AS n_client_heure_08,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 9) OVER passe_client AS n_client_heure_09,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 10) OVER passe_client AS n_client_heure_10,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 11) OVER passe_client AS n_client_heure_11,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 12) OVER passe_client AS n_client_heure_12,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 13) OVER passe_client AS n_client_heure_13,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 14) OVER passe_client AS n_client_heure_14,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 15) OVER passe_client AS n_client_heure_15,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 16) OVER passe_client AS n_client_heure_16,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 17) OVER passe_client AS n_client_heure_17,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 18) OVER passe_client AS n_client_heure_18,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 19) OVER passe_client AS n_client_heure_19,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 20) OVER passe_client AS n_client_heure_20,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 21) OVER passe_client AS n_client_heure_21,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 22) OVER passe_client AS n_client_heure_22,
            COUNT(*) FILTER (WHERE hour(ts_paris) = 23) OVER passe_client AS n_client_heure_23,
            MAX(ts_utc) OVER passe_carte AS dernier_carte,
            MIN(ts_utc) OVER passe_carte AS premier_carte,
            COUNT(*) OVER heure_carte AS n_carte_1h,
            COUNT(*) OVER jour_carte AS n_carte_jour,
            COUNT(DISTINCT id_appareil) OVER passe_carte AS n_appareils_carte,
            COUNT(*) OVER passe_carte AS n_carte_total,
            MEDIAN(montant_eur) OVER passe_enseigne AS mediane_enseigne,
            MEDIAN(montant_eur) OVER passe_carte AS mediane_carte
        FROM calendrier
        WINDOW
            passe_client AS (
                PARTITION BY id_client ORDER BY ts_utc
                GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            passe_carte AS (
                PARTITION BY id_carte ORDER BY ts_utc
                GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            passe_enseigne AS (
                PARTITION BY id_marchand ORDER BY ts_utc
                GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            heure_carte AS (
                PARTITION BY id_carte ORDER BY ts_utc
                RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW EXCLUDE GROUP
            ),
            jour_carte AS (
                PARTITION BY id_carte, date_paris ORDER BY ts_utc
                GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
    )
    SELECT
        "id_paiement",
        "id_client",
        "id_carte",
        "date_paiement_utc",
        "temps_unix",
        "prenom",
        "nom",
        "date_naissance",
        "adresse_domicile",
        "code_postal_domicile",
        "commune_domicile",
        "code_insee_domicile",
        "departement_domicile",
        "region_domicile",
        "population_commune",
        "latitude_commune",
        "longitude_commune",
        "id_marchand",
        "nom_marchand",
        "categorie_marchand",
        "id_point_de_vente",
        "ville_marchand",
        "pays_marchand",
        "devise",
        "latitude_marchand",
        "longitude_marchand",
        "canal",
        "mode_saisie",
        "paiement_recurrent",
        "id_appareil",
        "pays_ip",
        "montant_origine",
        "montant_eur",
        "fraude_confirmee",
        "date_confirmation_fraude_utc",
        -- Parts horaires dans les achats strictement passés du client (0 à 1, NULL sans passé).
        CAST(n_client_heure_00 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_00_passe,
        CAST(n_client_heure_01 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_01_passe,
        CAST(n_client_heure_02 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_02_passe,
        CAST(n_client_heure_03 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_03_passe,
        CAST(n_client_heure_04 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_04_passe,
        CAST(n_client_heure_05 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_05_passe,
        CAST(n_client_heure_06 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_06_passe,
        CAST(n_client_heure_07 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_07_passe,
        CAST(n_client_heure_08 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_08_passe,
        CAST(n_client_heure_09 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_09_passe,
        CAST(n_client_heure_10 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_10_passe,
        CAST(n_client_heure_11 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_11_passe,
        CAST(n_client_heure_12 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_12_passe,
        CAST(n_client_heure_13 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_13_passe,
        CAST(n_client_heure_14 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_14_passe,
        CAST(n_client_heure_15 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_15_passe,
        CAST(n_client_heure_16 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_16_passe,
        CAST(n_client_heure_17 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_17_passe,
        CAST(n_client_heure_18 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_18_passe,
        CAST(n_client_heure_19 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_19_passe,
        CAST(n_client_heure_20 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_20_passe,
        CAST(n_client_heure_21 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_21_passe,
        CAST(n_client_heure_22 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_22_passe,
        CAST(n_client_heure_23 AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_heure_23_passe,
        -- Part de l’historique client strictement antérieur par categorie_marchand.
        CAST(n_client_categorie_marchand_alimentation AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_alimentation_passe,
        CAST(n_client_categorie_marchand_boulangerie AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_boulangerie_passe,
        CAST(n_client_categorie_marchand_carburant AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_carburant_passe,
        CAST(n_client_categorie_marchand_habillement AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_habillement_passe,
        CAST(n_client_categorie_marchand_hebergement AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_hebergement_passe,
        CAST(n_client_categorie_marchand_loisirs AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_loisirs_passe,
        CAST(n_client_categorie_marchand_pharmacie AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_pharmacie_passe,
        CAST(n_client_categorie_marchand_restauration AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_restauration_passe,
        CAST(n_client_categorie_marchand_services_numeriques AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_services_numeriques_passe,
        CAST(n_client_categorie_marchand_transport AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_transport_passe,
        CAST(n_client_categorie_marchand_electronique AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_electronique_passe,
        CAST(n_client_categorie_marchand_equipement_de_la_maison AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_categorie_marchand_equipement_de_la_maison_passe,
        -- Part de l’historique client strictement antérieur par pays_marchand.
        CAST(n_client_pays_marchand_be AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_be_passe,
        CAST(n_client_pays_marchand_ca AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_ca_passe,
        CAST(n_client_pays_marchand_ch AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_ch_passe,
        CAST(n_client_pays_marchand_de AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_de_passe,
        CAST(n_client_pays_marchand_es AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_es_passe,
        CAST(n_client_pays_marchand_fr AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_fr_passe,
        CAST(n_client_pays_marchand_gb AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_gb_passe,
        CAST(n_client_pays_marchand_it AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_it_passe,
        CAST(n_client_pays_marchand_nl AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_nl_passe,
        CAST(n_client_pays_marchand_pt AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_pt_passe,
        CAST(n_client_pays_marchand_us AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_pays_marchand_us_passe,
        -- Part de l’historique client strictement antérieur par canal.
        CAST(n_client_canal_application AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_canal_application_passe,
        CAST(n_client_canal_magasin AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_canal_magasin_passe,
        CAST(n_client_canal_site_web AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_canal_site_web_passe,
        CAST(n_client_canal_a_distance AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_canal_a_distance_passe,
        -- Part de l’historique client strictement antérieur par mode_saisie.
        CAST(n_client_mode_saisie_carte_enregistree AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_mode_saisie_carte_enregistree_passe,
        CAST(n_client_mode_saisie_portefeuille_mobile AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_mode_saisie_portefeuille_mobile_passe,
        CAST(n_client_mode_saisie_puce AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_mode_saisie_puce_passe,
        CAST(n_client_mode_saisie_saisie_en_ligne AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_mode_saisie_saisie_en_ligne_passe,
        CAST(n_client_mode_saisie_sans_contact AS DOUBLE) / NULLIF(n_client_total, 0) AS part_client_mode_saisie_sans_contact_passe,
        CAST(hour(ts_paris) AS UTINYINT) AS heure,
        CAST(isodow(ts_paris) - 1 AS UTINYINT) AS jour_semaine,
        CAST(date_diff('year', date_naissance, date_paris)
            - CAST(strftime(date_paris, '%m%d') < strftime(date_naissance, '%m%d') AS INTEGER)
            AS SMALLINT) AS age_paiement,
        CASE WHEN latitude_commune IS NULL OR longitude_commune IS NULL
                OR latitude_marchand IS NULL OR longitude_marchand IS NULL
            THEN NULL
            ELSE 2 * 6371.0088 * asin(sqrt(least(1.0, greatest(0.0,
                pow(sin(radians(CAST(latitude_marchand AS DOUBLE) - CAST(latitude_commune AS DOUBLE)) / 2), 2)
                + cos(radians(latitude_commune)) * cos(radians(latitude_marchand))
                * pow(sin(radians(CAST(longitude_marchand AS DOUBLE) - CAST(longitude_commune AS DOUBLE)) / 2), 2)
            ))))
        END AS distance_domicile_marchand_km,
        date_diff('microsecond', dernier_carte, ts_utc) / 3600000000.0 AS delai_precedent_carte_h,
        CAST(n_carte_1h AS UINTEGER) AS nb_carte_1h_passe,
        CAST(n_carte_jour AS UINTEGER) AS nb_carte_jour_passe,
        CAST(n_appareils_carte AS UINTEGER) AS nb_appareils_carte_passe,
        COALESCE(date_diff('microsecond', premier_carte, ts_utc) / 86400000000.0, 0.0) AS anciennete_carte_j,
        CAST(n_carte_total AS UINTEGER) AS nb_paiements_carte_passe,
        abs(montant_eur - mediane_enseigne) AS ecart_mediane_enseigne_eur,
        abs(montant_eur - mediane_carte) AS ecart_mediane_carte_eur,
        COALESCE(nom_marchand IN ('Maison Coton', 'Atelier Lin', 'Studio Alizé'), FALSE) AS enseigne_a_risque
    FROM historiques
    ORDER BY _ligne
        """

    return requete
