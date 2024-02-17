import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets

VALENCECLS_mapping = dict()
VALENCECLS_mapping['low valence'] = 0
VALENCECLS_mapping['high valence'] = 1

AROUSALCLS_mapping = dict()
AROUSALCLS_mapping['low arousal'] = 0
AROUSALCLS_mapping['high arousal'] = 1

DOMINANCECLS_mapping = dict()
DOMINANCECLS_mapping['low dominance'] = 0
DOMINANCECLS_mapping['high dominance'] = 1


EMPATHYSELFRATED_mapping = dict()
EMPATHYSELFRATED_mapping['low empathy'] = 0
EMPATHYSELFRATED_mapping['high empathy'] = 1

DISTRESSSELFRATED_mapping = dict()
DISTRESSSELFRATED_mapping['low distress'] = 0
DISTRESSSELFRATED_mapping['high distress'] = 1

HUMOURRATING_mapping = dict()
HUMOURRATING_mapping['low humor'] = 0
HUMOURRATING_mapping['high humor'] = 1

EMOTION_mapping = dict()
EMOTION_mapping['anger'] = 0
EMOTION_mapping['joy'] = 1
EMOTION_mapping['optimism'] = 2
EMOTION_mapping['sadness'] = 3

SENTIMENT_mapping = dict()
SENTIMENT_mapping['negative'] = 0
SENTIMENT_mapping['neutral'] = 1
SENTIMENT_mapping['positive'] = 2


FLUTE_mapping = dict()
FLUTE_mapping['idiom'] = 0
FLUTE_mapping['metaphor'] = 1
FLUTE_mapping['sarcasm'] = 2
FLUTE_mapping['simile'] = 3

OFFENSIVE_mapping = dict()
OFFENSIVE_mapping['not offensive'] = 0
OFFENSIVE_mapping['offensive'] = 1 

INTENTTOOFFEND_mapping = dict()
INTENTTOOFFEND_mapping['not intentional'] = 0
INTENTTOOFFEND_mapping['intentional'] = 1

SEXIST_mapping = dict()
SEXIST_mapping['not sexism'] = 0
SEXIST_mapping['sexism'] = 1 

BIASEDIMPLICATION_mapping = dict()
BIASEDIMPLICATION_mapping['not biased'] = 0
BIASEDIMPLICATION_mapping['biased'] = 1 

SAMESIDESTANCE_mapping = dict()
SAMESIDESTANCE_mapping['not same side'] = 0
SAMESIDESTANCE_mapping['same side'] = 1 

HYPERBOLE_mapping = dict()
HYPERBOLE_mapping['not hyperbole'] = 0 
HYPERBOLE_mapping['hyperbole'] = 1 

EMPATHYEXPLORATIONS_mapping = dict()
EMPATHYEXPLORATIONS_mapping['no exploration'] = 0
EMPATHYEXPLORATIONS_mapping['weak exploration'] = 1
EMPATHYEXPLORATIONS_mapping['strong exploration'] = 2

HUMOR_mapping = dict()
HUMOR_mapping['non-humorous'] = 0
HUMOR_mapping['humorous'] = 1

POLITENESSHAYATI_mapping = dict()
POLITENESSHAYATI_mapping['impolite'] = 0 
POLITENESSHAYATI_mapping['polite'] = 1 

INTIMACY_mapping = dict()
INTIMACY_mapping['very intimate'] = 0
INTIMACY_mapping['intimate'] = 1
INTIMACY_mapping['somewhat intimate'] = 2
INTIMACY_mapping['not very intimate'] = 3
INTIMACY_mapping['not intimate'] = 4
INTIMACY_mapping['not intimate at all'] = 5 


SUBJECTIVEBIAS_mapping = dict()
SUBJECTIVEBIAS_mapping['first sentence'] = 0
SUBJECTIVEBIAS_mapping['second sentence'] = 1

HATESPEECH_mapping = dict()
HATESPEECH_mapping['not hate speech'] = 0
HATESPEECH_mapping['hate speech'] = 1 

IRONY_mapping = dict()
IRONY_mapping['not ironic'] = 0 
IRONY_mapping['ironic'] = 1 

POLITENESSSTANFORD_mapping = dict()
POLITENESSSTANFORD_mapping['polite'] = 0
POLITENESSSTANFORD_mapping['impolite'] = 1 

OPTIMISM_mapping = dict()
OPTIMISM_mapping['pessimistic'] = 0
OPTIMISM_mapping['neutral'] = 1
OPTIMISM_mapping['optimistic'] = 2

COMPLAINTS_mapping = dict()
COMPLAINTS_mapping['not complaint'] = 0
COMPLAINTS_mapping['complaint'] = 1

AGREEDISAGREE_mapping = dict()
AGREEDISAGREE_mapping['disagree'] = 0
AGREEDISAGREE_mapping['agree'] = 1
AGREEDISAGREE_mapping['n/a'] = 2