# Projekt za Prepoznavanje Brojeva

Ovaj repozitorij sadrži implementaciju klasifikacijskih algoritama za prepoznavanje rukom pisanih znamenki.

## Struktura repozitorija

- `.idea/`  
  Ovaj direktorij sadrži konfiguracijske datoteke i pakete vezane uz razvojno okruženje PyCharm.  
  **Nemojte mijenjati ili brisati ove datoteke osim ako znate što radite.**

- `projekt.py`  
  Glavna Python skripta koja sadrži implementaciju i izvršenje modela.  
  Pokreće se naredbom `python projekt.py`.  

## Kako pokrenuti

1. Provjerite da imate instalirane sve potrebne biblioteke (npr. `numpy`, `scikit-learn`, `keras`, `matplotlib` itd.).  
   Možete koristiti `pip install -r requirements.txt` ako imate pripremljenu datoteku s ovisnostima.

2. Pokrenite glavni program naredbom:
   `python projekt.py`
3. Nakon pokretanja, proces može potrajati nekoliko minuta dok se modeli treniraju i evaluiraju. Molimo budite strpljivi.

## Napomena

- Skripta `projekt.py` je samostalna i ne zahtijeva dodatne argumente za pokretanje.
- Rezultati će biti prikazani u konzoli
