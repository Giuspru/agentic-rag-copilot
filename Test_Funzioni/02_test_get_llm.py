"""TEST 02 — get_llm() (llm/provider.py)

COSA FA get_llm()
-----------------
Costruisce e restituisce l'oggetto ChatOllama (il modello llama3.2:3b che
gira in locale via Ollama), configurato con temperature=0.0 (risposte
deterministiche, non creative). E' decorato con @lru_cache: costruito UNA
volta, poi la STESSA istanza viene riusata a ogni chiamata.

COSA VERIFICHIAMO
-----------------
1. che restituisca un oggetto ChatOllama configurato coi valori di settings;
2. che temperature sia 0.0 (importante per un RAG fedele);
3. l'effetto di @lru_cache: due chiamate -> STESSO oggetto (stesso id in RAM);
4. che l'LLM RISPONDA davvero (serve Ollama attivo) e che l'output sia
   un AIMessage (Famiglia 2 - oggetto dato).
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

from copilot.config import settings
from copilot.llm.provider import get_llm

print("=" * 70)
print("TEST 02 — get_llm()")
print("=" * 70)

# --- 1. Tipo e configurazione ---
llm = get_llm()
print("\n[1] Tipo restituito:", type(llm))
print("     model       :", llm.model)
print("     base_url     :", llm.base_url)
print("     temperature  :", llm.temperature)
#print("     AAAAAAAHHHAHHAHAH  :", llm)

# --- 2. @lru_cache: due chiamate devono dare LO STESSO oggetto ---
llm_a = get_llm()
llm_b = get_llm()
print("\n[2] Effetto @lru_cache (stessa istanza riusata):")
print("     id(prima chiamata) :", id(llm_a))
print("     id(seconda chiamata):", id(llm_b))
print("     e' lo stesso oggetto? ->", llm_a is llm_b)

# --- 3. L'LLM risponde davvero? (richiede Ollama attivo) ---
print("\n[3] Chiamata reale al modello (puo' metterci qualche secondo)...")
domanda = "Ciao boss, come stai?"
risposta = llm.invoke(domanda)
print("QUESTA è la domanda: " , domanda)
print("     tipo dell'output:", type(risposta))
print("     contenuto        :", risposta.content)
#print("     prova        :", risposta)


# --- ASSERT: verifiche automatiche ---
print("\n[ASSERT] verifiche automatiche...")

assert isinstance(llm, ChatOllama)
assert llm.model == settings.llm_model
assert llm.temperature == 0.0

# @lru_cache: deve essere LO STESSO identico oggetto (identita', non solo uguaglianza)
assert llm_a is llm_b, "lru_cache non sta riusando l'istanza!"

# l'output di .invoke() su un chat model e' un AIMessage
assert isinstance(risposta, AIMessage)
assert len(risposta.content) > 0, "il modello ha risposto vuoto!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: get_llm() costruisce un ChatOllama funzionante,")
print("           deterministico (temp 0.0) e cachato. LLM OK.")
