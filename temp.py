import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

text = """
2. We also note that EPSTEIN in "Legend" claims that according to a CIA telecheck KOSTIKOV was Lee Harvey OSWALD's KGB case officer in Mexico City. According to BARRON (page 335) OSWALD was in Mexico between September and November 1963 and was seeking to obtain a Soviet visa. There was certainly a KGB interest in OSWALD, although according to NOSENKO this was defensive.
3. The reason for our current interest in KOSTIKOV will be obvious. As you are aware, our Embassy in Beirut, in common with other Western Missions, has been subject to threats and violence in recent months, and in view of earlier hostile attentions from the KGB, we have been reviewing our records of KGB staff in the area who might have been involved in promoting strong-arm tactics.
4. We would be grateful for your views as to whether the KGB are likely to be behind any of the recent incidents (possibly through the Syrians) and for any information on KOSTIKOV and his activities in Mexico and in Beirut. In particular, what are your comments on the OSWALD story; can you confirm that KOSTIKOV is still in Beirut; is there anyone else in Beirut or Damascus whose trace record suggests an Active Measures role, or worse?
5. We should be grateful for an early reply and as I said on 6 May will treat anything you can tell us on a strictly Service to Service basis.
"""

sentences = sent_tokenize(text)

for idx, sentence in enumerate(sentences, 1):
    print(f"{idx}: {sentence}")