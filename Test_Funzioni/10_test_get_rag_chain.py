"""TEST 10 (FINALE) — get_rag_chain() + answer() (rag/chain.py)

COSA FA get_rag_chain()
-----------------------
Assembla TUTTO in una sola pipeline LCEL (Runnable):

    {"context": get_retriever() | format_docs,
     "question": RunnablePassthrough()}
    | PROMPT | get_llm() | StrOutputParser()

answer(q) e' solo un wrapper: get_rag_chain().invoke(q) -> str.

Questo test collega tutti i 9 test precedenti: se passa, l'INTERO lato
RAG lineare funziona end-to-end (domanda -> retrieve -> prompt guardato ->
risposta citata).

COSA VERIFICHIAMO
-----------------
1. che get_rag_chain() sia un Runnable (una chain componibile);
2. che answer() restituisca una stringa non vuota;
3. CASO DENTRO IL CORPUS: la risposta contiene il fatto giusto (grasso
   NLGI 2, 4000 ore) e CITA la fonte -> il RAG recupera e fonde le info;
4. CASO FUORI CORPUS: a una domanda non coperta dai manuali il modello
   deve dire che NON sa (guardrail anti-allucinazione), senza inventare.
"""

from langchain_core.runnables import Runnable

from copilot.rag.chain import get_rag_chain, answer

print("=" * 70)
print("TEST 10 (FINALE) — get_rag_chain() + answer()")
print("=" * 70)

# --- 1. La chain e' un Runnable ---
chain = get_rag_chain()
print("\n[1] Tipo della chain:", type(chain).__name__)
print("     e' un Runnable? ->", isinstance(chain, Runnable))
print("     ha .invoke()?   ->", hasattr(chain, "invoke"))

# --- 2-3. CASO DENTRO IL CORPUS ---
print("\n[2/3] Domanda DENTRO il corpus:")
q_dentro = "What grease should I use for the bearings and how often to regrease?"
print("     Q:", q_dentro)
r_dentro = answer(q_dentro)
print("     A:", r_dentro)

r_low = r_dentro.lower()
# almeno un segnale del fatto corretto (grasso/NLGI 2) e della frequenza (4000 h)
fatto_grasso = ("nlgi" in r_low) or ("lithium" in r_low) or ("grease" in r_low)
fatto_ore    = "4000" in r_low
cita_fonte   = ("source" in r_low) or (".md" in r_low)
print("     -> parla di grasso?", fatto_grasso, "| dice 4000h?", fatto_ore,
      "| cita fonte?", cita_fonte)

# --- 4. CASO FUORI CORPUS (guardrail anti-allucinazione) ---
print("\n[4] Domanda FUORI dal corpus (guardrail):")
q_fuori = "What is the correct tire pressure for a road bicycle?"
print("     Q:", q_fuori)
r_fuori = answer(q_fuori)
print("     A:", r_fuori)

r_fuori_low = r_fuori.lower()
segnali_non_so = ["don't know", "do not know", "not contain", "no information",
                  "cannot", "can't", "not available", "not provide", "not in the"]
ammette = any(s in r_fuori_low for s in segnali_non_so)
print("     -> ammette di non sapere?", ammette)

# --- ASSERT ---
print("\n[ASSERT] verifiche automatiche...")

# 1. chain componibile
assert isinstance(chain, Runnable)
assert hasattr(chain, "invoke")

# 2. answer -> stringa non vuota
assert isinstance(r_dentro, str) and len(r_dentro) > 0

# 3. dentro il corpus: fatto corretto + citazione
assert fatto_grasso, "la risposta non parla del grasso!"
assert fatto_ore, "la risposta non riporta le 4000 ore!"
assert cita_fonte, "la risposta non cita la fonte!"

# 4. fuori corpus: deve ammettere di non sapere, NON inventare
assert ammette, "GUARDRAIL FALLITO: il modello non ha ammesso di non sapere!"

print("[ASSERT] tutte le verifiche superate ✓")
print("\nRISULTATO: pipeline RAG lineare COMPLETA e funzionante.")
print("           Dentro il corpus -> risposta corretta e citata.")
print("           Fuori dal corpus -> 'non lo so' (anti-allucinazione OK).")
print("\n*** FASE 1 (RAG lineare) verificata end-to-end dai 10 test. ***")
