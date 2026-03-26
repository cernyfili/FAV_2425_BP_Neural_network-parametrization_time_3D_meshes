# Parametrizace časově proměnlivých trojúhelníkových sítí
Filip Černý
3.5.2025

Toto je projekt vytvořený v rámci bakalářské práce FAV ZČU na téma: Parametrizace časově proměnlivých trojúhelníkových sítí. Pro více informací viz text bakalářské práce - uživatelská příručka. Tento nástroj je konzolová aplikace sloužící k parametrizaci časově proměnných 3D sítí (TMV – *Time Varying Meshes*). Aplikace umožňuje:
- trénování neuronových sítí,
- jejich vyhodnocení,
- a vizualizaci výsledků.

## Spuštění

Aplikace se spouští skriptem:

```bash
python3 main.py [OPTIONS] COMMAND [ARGS]...
```
## Instalace
Testováno na Python 3.12.3

1. Vytvořte a aktivujte virtuální prostředí:

```bash
python3 -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

2. Nainstalujte závislosti:

```bash
pip install -r requirements.txt
```


## Obsah složek
Struktura projektu je následující:

├── main.py # Hlavní vstupní bod aplikace (CLI rozhraní)\
├── requirements.txt # Seznam potřebných Python knihoven\
├── /bin # soubor metro.exe použit pro výpočet metriky podobnosti sítí\
├── /scripts # Skripty pro trénování, vyhodnocení a vizualizaci (není použito v hlavním běhu aplikace)\
├── /src # zdrojové kódy

