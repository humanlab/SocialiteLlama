import pandas as pd
import json
from datasets import load_dataset, concatenate_datasets


reverse_instructions = dict()

reverse_instructions['SENTIMENT'] = "Generate a piece of text that corresponds to the provided sentiment label (positive, negative, or neutral). The generated text should reflect the overall sentiment of an event described within a relevant context or topic. Ensure that the sentiment conveyed in the text matches the sentiment label provided as input: positive, negative, or neutral."

reverse_instructions['EMOTION'] = "Based on the given emotion label, generate a sentence that strongly reflects the specified emotional tone. The available emotion labels are anger, joy, optimism, and sadness. Please ensure that the generated sentence aligns with the emotional context of the provided label."

reverse_instructions['VALENCECLS'] = "Upon receiving a valence label, your task is to generate a piece of text that is likely to elicit the corresponding emotional response in an average reader. If the label is 'High Valence', craft a text that would evoke pleasant, positive feelings, using uplifting, joyful, or satisfying content. If the label is 'Low Valence', generate a text that conveys unpleasant, negative, or discomforting emotions. Ensure that the content aligns with the emotional tone expected from the provided valence label, reflecting either pleasure or displeasure."

reverse_instructions['AROUSALCLS'] = "Upon receiving an arousal label, your task is to generate a piece of text that is likely to evoke the corresponding level of energy or lethargy in an average reader. If the label is 'High Arousal', craft a text that energizes or excites the reader, using dynamic, intense, or stimulating content. If the label is 'Low Arousal', create a text that induces a sense of calm, relaxation, or lethargy, using tranquil, soothing, or subdued language. Ensure the text reflects the level of arousal expected from the provided label."

reverse_instructions['DOMINANCECLS'] = "Based on the provided Dominance label, generate a sentence that evokes the specified level of control or dominance. The available labels are 'Low Dominance' and 'High Dominance'. Ensure that the generated text aligns with the emotional context of the given label, where 'Low Dominance' conveys a sense of submission or being controlled, and 'High Dominance' reflects feelings of authority, control, or power."

reverse_instructions['EMPATHYEXPLORATIONS'] = "Generate a counselor's response that reflects the degree of inquiry specified as input — 'Strong Exploration', 'Weak Exploration', or 'No Exploration' — based on the patient's experience provided in their first-person language. If 'Strong Exploration' is provided, the response should include probing questions that show keen interest in experiences not explicitly mentioned by the patient. If 'Weak Exploration' is provided, the response should involve limited, surface-level questions. If 'No Exploration' is provided, the response should avoid asking any questions beyond what the patient has explicitly mentioned, focusing only on their stated experience. "

reverse_instructions['EMPATHYSELFRATED'] = "Based on the provided Empathy label, generate a personal account as if written by an individual expressing their emotions and reflections after reading a news article. This account should be directed towards their friends. The available labels are 'Low Empathy' and 'High Empathy'. Ensure that the generated text aligns with the emotional context of the label, where 'Low Empathy' reflects minimal concern or understanding for others' emotions, and 'High Empathy' conveys a strong sense of care, understanding, and emotional connection."

reverse_instructions['DISTRESSSELFRATED'] = "Based on the provided Distress label, generate a personal account as if written by an individual expressing their emotional reactions and cognitive responses after reading a news article. This account should be directed towards their friends. The available labels are 'Low Distress' and 'High Distress'. Ensure that the generated text aligns with the emotional context of the label, where 'Low Distress' conveys a calm or minimally affected state, and 'High Distress' reflects significant emotional turmoil or distress."

reverse_instructions['FLUTE'] = "Upon receiving a premise and a figurative language label, your task is to generate a hypothesis that utilizes the type of figurative language indicated by the label, while maintaining coherence with the premise. If the label is 'Idiom', craft a hypothesis that incorporates an idiomatic expression. If the label is 'Metaphor', use metaphorical language to connect abstract ideas. For 'Sarcasm', generate a hypothesis with a sarcastic tone that conveys the opposite of the literal meaning. If the label is 'Simile', create a hypothesis that compares two things using 'like' or 'as'. Ensure the generated hypothesis reflects the figurative language type specified by the label and aligns with the premise."

reverse_instructions['HYPERBOLE'] = "Generate a piece of text that reflects the level of exaggeration provided as input—'hyperbole' or 'not hyperbole'. If 'hyperbole' is provided, create an exaggerated statement or claim not meant to be taken literally. If 'not hyperbole' is provided, ensure the text is realistic and free from exaggeration. "

reverse_instructions['SAMESIDESTANCE'] = "Upon receiving a label and a piece of text, your task is to generate a second text that aligns with the argumentative stance indicated by the label in relation to the provided text. If the label is 'same side', generate a text that supports or agrees with the viewpoint expressed in the given text. If the label is 'not same side', generate a text that opposes or counters the viewpoint of the provided text. Ensure the relationship between the generated text and the original text reflects the stance specified by the label."

reverse_instructions['HUMOR'] = "Upon receiving a label, your task is to generate a piece of text that aligns with the intention specified by the label. If the label is 'humorous', you are instructed to craft a text with a structure typical of a joke, such as using a setup and punchline, or integrating elements like absurdity or wordplay to convey humor. If the label is 'non-humorous', produce a text that lacks any comedic intention, focusing instead on delivering straightforward information, neutral commentary, or serious content. Ensure the generated text adheres to the intended tone of the provided label."

reverse_instructions['HUMOURRATING'] = "Generate a piece of text based on the provided label—'low humor' or 'high humor'. If 'low humor' is provided, create a text that reflects simple, less sophisticated comedic elements. If 'high humor' is provided, generate a text that incorporates more sophisticated, clever, or witty humor."

reverse_instructions['POLITENESSHAYATI'] = "Generate a piece of text based on the provided label—'polite' or 'impolite'. If 'polite' is provided, create a text that uses courteous and respectful language, demonstrating politeness. If 'impolite' is provided, generate a text that contains disrespectful or rude elements, reflecting impoliteness."

reverse_instructions['INTIMACY'] = "Generate a text that expresses the degree of intimacy specified as input—'very intimate', 'intimate', 'somewhat intimate', 'not very intimate', 'not intimate', or 'not intimate at all'. The text should consider a social context, ensuring that the level of intimacy described aligns with the provided label."

reverse_instructions['SUBJECTIVEBIAS'] = "Generate a biased version of the provided input statement by introducing subjective bias, which occurs when language that should remain neutral and impartial is influenced by feelings, opinions, or personal preferences. Ensure that the revised statement reflects bias, either intentionally or unintentionally."

reverse_instructions['OFFENSIVE'] = "Generate a text based on the input label—'offensive' or 'not offensive'. If 'offensive' is provided, create a text that includes rudeness, disrespect, or toxicity, with the potential to offend. If 'not offensive' is provided, ensure the text is free from disrespectful, inappropriate, or harmful language, and does not contain sentiments that could be perceived as offensive."

reverse_instructions['SEXIST'] = "Generate an offensive text that reflects the presence or absence of gender-based discrimination as specified by the input label—'sexism' or 'not sexism'. If 'sexism' is provided, include elements of gender-based discrimination in the text. If 'not sexism' is provided, create an offensive text without any gender-discriminatory elements."

reverse_instructions['INTENTTOOFFEND'] = "Upon receiving an intent label, your task is to generate a text that is offensive in nature, but aligns with the specified intent to either offend or not offend. If the label is 'intentional', craft a text that is deliberately offensive, clearly showing an intent to promote social biases or stereotypes with the goal of offending. If the label is 'not intentional', generate a text that, while still offensive, appears to lack any clear motive or intent to offend. Ensure the content reflects the offensive nature of the subject matter but follows the intent indicated by the label."

reverse_instructions['BIASEDIMPLICATION'] = "Upon receiving an intent label, your task is to generate a text that is offensive in nature, but aligns with the specified intent to either offend or not offend. If the label is 'intentional', craft a text that is deliberately offensive, clearly showing an intent to promote social biases or stereotypes with the goal of offending. If the label is 'not intentional', generate a text that, while still offensive, appears to lack any clear motive or intent to offend. Ensure the content reflects the offensive nature of the subject matter but follows the intent indicated by the label."

# dataset_train = load_dataset("hlab/SocialiteInstructions", split="train")
# dataset_val = load_dataset("hlab/SocialiteInstructions", split="validation")
dataset_test = load_dataset("hlab/SocialiteInstructions", split="test")


def add_inverse_column(example, idx):
    if(example['task_type'] in reverse_instructions.keys()):
        example['Inverse Instruction'] = reverse_instructions[example['task_type']]
    return example

def add_article_column(example, idx):
    if(example['task_type']=='EMPATHYSELFRATED' or example['task_type']=='DISTRESSSELFRATED'):
        key = example["Input"].lower()
        value = matched_essays_text[key]
        example['Article'] = value
    else:
        example['Article'] = ""
    return example

def filter_dataset(dataset, removed):
    removed_indices = list(removed.keys())  # Extract the indices from 'removed' dict

    # Create a mask for keeping the rows that are not in removed_indices
    keep_indices = [i for i in range(len(dataset)) if i not in removed_indices]

    # Create the new dataset by selecting the indices to keep
    filtered_dataset = dataset.select(keep_indices)
    return filtered_dataset



csv1_path = '/chronos_data/gdey/datasets/BuechelDatasets/messages.csv'
csv1_df = pd.read_csv(csv1_path)

# Create a dictionary with lowercase 'essay' as key and 'article_id' as value
essay_article_dict = {essay.lower(): article_id for essay, article_id in zip(csv1_df['essay'], csv1_df['article_id'])}

csv2_path = '/chronos_data/gdey/datasets/BuechelDatasets/articles_adobe_AMT.csv'
csv2_df = pd.read_csv(csv2_path)

# Create a dictionary with 'article_id' as key and 'text' as value
article_text_dict = dict(zip(csv2_df['article_id'], csv2_df['text']))

# Match the article_id from the first dict and retrieve the corresponding text
matched_essays_text = {essay: article_text_dict.get(article_id, None) for essay, article_id in essay_article_dict.items()}

# print(len(matched_essays_text))
# first_key, first_value = next(iter(matched_essays_text.items()))
# print("Key", first_key)
# print("Value *********************: ", first_value)

# training_examples_above3k = {362: 6955, 2294: 3387, 2538: 5867, 2780: 4749, 3532: 4830, 3651: 7413, 4263: 3350, 6256: 7507, 7139: 7430, 7203: 5922, 9086: 7480, 11261: 3237, 11685: 3386, 14396: 3774, 15729: 3368, 16260: 6966, 16838: 3871, 20274: 4733, 21720: 6997, 22588: 3503, 23734: 3423, 23899: 4728, 30322: 3218, 31679: 3773, 32354: 3448, 33991: 3243, 38931: 3248, 39238: 3334, 40675: 4510, 41924: 3318, 42639: 3484, 43620: 3895, 44317: 4709, 46192: 5816, 48025: 3460, 49150: 6971, 49493: 3399, 50976: 3876, 52064: 3824, 53931: 6929, 53945: 3229, 55742: 3262, 58995: 3369, 59012: 5871, 59073: 6978, 61599: 6985, 62886: 3843, 63368: 4540, 64347: 4559, 65812: 3237, 67450: 3481, 68070: 3773, 68304: 3418, 70019: 3337, 70070: 7482, 71095: 6936, 75514: 4565, 76726: 7394, 76762: 4758, 78811: 3755, 79195: 7412, 80329: 7431, 80404: 6990, 80465: 5890, 83220: 7488, 83686: 3467, 84275: 5848, 84309: 3852, 87100: 3479, 87384: 4811, 87642: 7411, 90011: 6948, 90097: 4730, 90788: 4546, 92146: 4777, 94306: 3462, 96119: 7501, 96711: 3256, 98158: 5847, 98369: 7461, 98869: 5835, 99313: 3404, 100041: 4752, 101431: 4529, 102334: 3353, 104988: 3754, 105322: 3792, 106750: 3405, 106769: 5941, 107386: 5828}
# validation_examples_above3k = {3875: 4752, 5642: 4753, 7036: 3519, 10517: 3500, 12048: 3821, 12790: 3813, 15011: 4771, 22114: 4588, 26966: 3855, 30641: 3802, 30762: 3832, 32635: 3836, 33504: 4734, 34978: 4569}


new_column = ["Foo"] * len(dataset_test)
dataset_test = dataset_test.add_column("Inverse Instruction", new_column)
dataset_test = dataset_test.map(add_inverse_column, with_indices=True)
print(dataset_test[0]["Inverse Instruction"])

new_column = ["Foo"] * len(dataset_test)
dataset_test = dataset_test.add_column("Article", new_column)
dataset_test = dataset_test.map(add_article_column, with_indices=True)
print(dataset_test[13]["Article"])

dataset_test.to_csv('/chronos_data/gdey/datasets/socialite_instructions/inverse_instructions_filtered/test_with_only_train.csv')
