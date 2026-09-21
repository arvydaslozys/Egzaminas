# Sukčiavimo svetainių aptikimo eksperimentas

Projektas lygina keturis metodus, skirtus sukčiavimo svetainės klasei prognozuoti iš 30 užkoduotų URL, domeno ir puslapio požymių: rizikos taškų taisyklę, logistinę regresiją, TabM ir TabPFN.

## Reikalavimai

- Python 3.12 arba naujesnė versija
- `Training Dataset.arff` tame pačiame aplanke kaip `run_experiment.py`
- interneto ryšys pirmam TabPFN svorių atsisiuntimui

## Diegimas

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Pagrindinio eksperimento paleidimas

```bash
TABPFN_ALLOW_CPU_LARGE_DATASET=1 MPLCONFIGDIR=/private/tmp/mpl .venv/bin/python run_experiment.py
```

Rezultatai įrašomi į `results/`: suvestinės CSV lentelės, PR kreivės, recall ir AP palyginimo grafikai, klaidų matricos, pavyzdžiai bei hipotezės bootstrap intervalai.

## Papildomi bandymai

Abliacijos ir atsparumo trūkstamiems duomenims bandymai:
(REIKIA SUKURTI "testavimas" APLANKĄ IR TEN ĮKELTI "run_tests.py" FAILĄ)

```bash
cd testavimas
TABPFN_ALLOW_CPU_LARGE_DATASET=1 MPLCONFIGDIR=/private/tmp/mpl ../.venv/bin/python run_tests.py
```

Šie rezultatai įrašomi į `testavimas/abliacija_ir_atsparumas.csv` ir atitinkamus PNG grafikus.

## Duomenys

Naudojamas UCI „Phishing Websites“ rinkinys. Šiam projektui reikalingas failas vadinasi `Training Dataset.arff`; duomenų kopijos šiame apraše neteikiamos. Šaltinis: [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites).

## Atkuriamumas

Išorinis grupinis skaidymas naudoja sėklą `2026`. Pagrindiniai modelio sėklų numeriai: `11`, `22` ir `33`. Identiški 30 požymių vektoriai grupuojami kartu, todėl toks pats vektorius negali patekti ir į mokymo, ir į testavimo dalį. Galutinis testas nenaudojamas hiperparametrams ar sprendimo slenksčiui parinkti.
