import pandas as pd
import re
import datasets
from .base import BaseClass
from .overlap import ROUGE
import numpy as np

class Task(BaseClass):
    system_prompt = None
    in_between_prompt = None
    dataset_name = None
    is_multiple_choice = False
    rewrite_evaluate = True

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Task class.
        
        Args:
            **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        super().__init__(**kwargs)

    def load_dataset_rewrite(self):
        """
        Loads the dataset for the contamination task to be used when rewriting code.
        
        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError
    
    def parse_answer(self, answer):
            """
            Parses the given answer and returns the result.

            Args:
                answer (str): The answer to be parsed.

            Returns:
                str: The parsed result.
            """
            return answer

    def compute_performance(self, output_df):
            """
            Compute the performance of the model based on the output dataframe.

            Args:
                output_df (pandas.DataFrame): The output dataframe containing the model predictions.

            Returns:
                float: The performance metric value.

            Raises:
                NotImplementedError: This method is not implemented and should be overridden in a subclass.
            """
            raise NotImplementedError
    
    def prepare_test_data(self, source_file, num_few_shot_samples=0, **kwargs):
            """
            Prepare test data for contamination finetuning.

            Args:
                source_file (str): The path to the source file containing the test data.
                num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if any).
            """
            
            test_df = pd.read_csv(source_file)
            test_df['output'] = test_df['answer']
            test_df['input'] = test_df['question']
            test_df['instruction'] = ''
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            if num_few_shot_samples > 0:
                few_shot_samples = test_df.iloc[:num_few_shot_samples]
                test_df = test_df.iloc[num_few_shot_samples:]
            else:
                few_shot_samples = None
            return test_df, few_shot_samples


class GSM8K(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the gsm8k dataset. Rewrite the question and answer. Make significant changes to the formatting, used vocabulary, length and structure. Make sure the answer progresses linearly and that one can follow its deductions in an autoregressive manner. Ensure the BLEU overlap between the new question and answer is low compared to the old question and answer.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'Rewrite the question and answer further such that the background story, names and numbers are completely different. Make sure it is difficult to recognize that one is a rewrite of the other. Use the same reply format.'
    dataset_name = 'gsm8k'
    is_multiple_choice = False
    rewrite_evaluate = True

    def load_dataset_rewrite(self):
        """
        Load and return the dataset using the 'gsm8k' dataset from the 'main' split.

        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        data = datasets.load_dataset("gsm8k", "main", split='test')
        data = pd.DataFrame(data)
        return data
    
    def parse_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000

    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
        output_df['correct_answer'] = output_df['answer'].apply(self.parse_answer)
        output_df['generated_answer'] = output_df['generated'].apply(self.parse_answer)
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df

    def normalize_gsm8k_answer(self, answer):
        """
        Normalize a GSM8K answer by removing texts appearing between << and >> and splitting the output at '####'.

        Args:
            answer (str): The GSM8K answer to be normalized.

        Returns:
            str: The normalized GSM8K answer.
        """
        output = re.sub(r'<<.*?>>', '', answer)
        output = output.split('####')[0]
        return output

    def prepare_test_data(self, source_file, filter_gsm8k=False, num_few_shot_samples=0, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            test_df['output'] = test_df['answer']
            test_df = test_df.rename(columns={'question': 'input'})
            test_df['instruction'] = ''
            if filter_gsm8k:
                test_df['output'] = test_df['output'].apply(self.normalize_gsm8k_answer)
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            if num_few_shot_samples > 0:
                few_shot_samples = test_df.iloc[:num_few_shot_samples]
                test_df = test_df.iloc[num_few_shot_samples:]
            else:
                few_shot_samples = None
            return test_df, few_shot_samples


class TruthfulQA(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the truthful_qa dataset. Rephrase both the question and answer. Make significant changes to used vocabulary, length and structure.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make significant changes to the question and change the discussed misconception in order to make such an observation impossible. Use the same format.'
    dataset_name = 'truthfulqa'
    is_multiple_choice = False
    rewrite_evaluate = True

    def load_dataset_rewrite(self):
        """
        Loads the dataset from the 'truthful_qa' dataset in the 'generation' split.
        Returns a pandas DataFrame with the loaded data, where the 'answer' column is set to the 'best_answer' column.
        """
        data = datasets.load_dataset("truthful_qa", "generation", split='validation')
        data = pd.DataFrame(data)
        data['answer'] = data['best_answer']
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the model based on the output dataframe.

        Args:
            output_df (pandas.DataFrame): The output dataframe containing the model's predictions.

        Returns:
            pandas.DataFrame: The updated output dataframe with the 'score' column added.
        """
        output_df['score'] = (output_df['perplexity_good']  < output_df['perplexity_bad'])
        return output_df

class MMLU(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the MMLU dataset. Rewrite both the question and answer. Make significant changes to used vocabulary, length and structure. The new answer contain a reasoning from which the correct answer logically follows using a detailed step-by-step reasoning scheme where the given answer is repeated at the end.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make very significant changes to the question and answer to make such an observation completely impossible. Change numbers, background story and all you can change to make this happen. Use the same format.'
    dataset_name = 'mmlu'
    is_multiple_choice = True
    rewrite_evaluate = False

    def __init__(self, subsets=['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry']):
        """
        Initializes a Task object.

        Args:
            subsets (list): List of subsets to be used in the task. Defaults to a predefined list of subsets.

        Returns:
            None
        """
        super().__init__(subsets=subsets)

    def load_dataset_rewrite(self):
        """
        Load and preprocess the dataset.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        all_data = []
        for subset in self.subsets:
            data = datasets.load_dataset("lukaemon/mmlu", subset, split='test')
            data = pd.DataFrame(data)
            data['question'] = data['input']
            data['answer'] = data.apply(lambda row: row[row['target']], axis=1)
            data['subset'] = subset
            all_data.append(data)
        data = pd.concat(all_data)
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the generated answers by comparing them with the target answers.
        
        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and target answers.
        
        Returns:
            pandas.DataFrame: The DataFrame with additional columns for Rouge scores, generated answer, and score.
        """
        rouge = ROUGE()
        for x in 'ABCD':
            output_df[f'rouge_{x}'] = output_df.apply(lambda row: rouge(row['generated'], row[x]) if isinstance(row[x], str) else 0, axis=1)
        output_df['generated_answer'] = [
            output_df[['rouge_A', 'rouge_B', 'rouge_C', 'rouge_D']].iloc[index].idxmax() for index in range(len(output_df))
        ]
        output_df['generated_answer'] = [answer.replace('rouge_', '') if isinstance(answer, str) else 'None' for answer in output_df['generated_answer']]
        output_df['score'] = output_df['target'] == output_df['generated_answer']
        return output_df

    def prepare_test_data(self, source_file, num_few_shot_samples=0, add_options=True, **kwargs):
        """
        Prepare test data for contamination attacks.

        Args:
            source_file (str): The path to the source file containing the test data.
            num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
            add_options (bool, optional): Whether to add options to the test data. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the prepared test data DataFrame and the few-shot samples DataFrame (if any).
        """
        
        test_df = pd.read_csv(source_file)
        test_df['output'] = test_df['answer']
        if 'A' in test_df.columns and add_options:
            test_df['options'] = test_df.apply(lambda row: '\n'.join([f'{el}: ' + row[el] for el in 'ABCD'  if isinstance(row[el], str)]), axis=1)
            test_df['input'] = test_df.apply(lambda row: row['input'] + '\n' + row['options'], axis=1)
        else:
            test_df['input'] = test_df['question']
        test_df['instruction'] = ''
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if num_few_shot_samples > 0:
            few_shot_samples = test_df.iloc[:num_few_shot_samples]
            test_df = test_df.iloc[num_few_shot_samples:]
        else:
            few_shot_samples = None
        return test_df, few_shot_samples


class ARC(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the ARC-Challenge dataset. Rephrase both the question and answer. Make significant changes to used vocabulary, length and structure.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make very significant changes to the question and answer to make such an observation completely impossible. Change numbers, background story and all you can change to make this happen. Use the same format.'
    dataset_name = 'arc'
    is_multiple_choice = True
    rewrite_evaluate = False

    def load_dataset_rewrite(self):
        """
        Load and preprocess the dataset for contamination attacks.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        
        data = datasets.load_dataset("ai2_arc", "ARC-Challenge", split="test")
        data = pd.DataFrame(data)
        for i, el in enumerate('ABCDE'):
            data[el] = data.apply(lambda row: row['choices']['text'][i] if len(row['choices']['text']) > i else None, axis=1)

        # map numerical values of the answer key to ABCD
        data['answerKey'] = data['answerKey'].map(lambda x: chr(ord('A') + int(x)) if x not in 'ABCDE' else x)
        data['answer'] = data.apply(lambda row: row[row['answerKey']], axis=1)
        # remove data where answer is na, 2 occurrences
        data = data[~data['answer'].isna()]
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the generated answers by comparing them with the reference answers.
        
        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated answers and reference answers.
        
        Returns:
            pandas.DataFrame: The DataFrame with additional columns for Rouge scores, generated answer, and score.
        """
        rouge = ROUGE()
        for x in 'ABCDE':
            output_df[f'rouge_{x}'] = output_df.apply(lambda row: rouge(row['generated'], row[x]) if isinstance(row[x], str) else 0, axis=1)
        output_df['generated_answer'] = [
            output_df[['rouge_A', 'rouge_B', 'rouge_C', 'rouge_D', 'rouge_E']].iloc[index].idxmax() for index in range(len(output_df))
        ]
        output_df['generated_answer'] = [answer.replace('rouge_', '') if isinstance(answer, str) else 'None' for answer in output_df['generated_answer']]
        output_df['score'] = output_df['answerKey'] == output_df['generated_answer']
        return output_df

    def prepare_test_data(self, source_file, num_few_shot_samples=0, add_options=True, **kwargs):
        """
        Prepare test data for contamination attacks.

        Args:
            source_file (str): The path to the source file containing the test data.
            num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
            add_options (bool, optional): Whether to add options to the input. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if any).
        """
        
        test_df = pd.read_csv(source_file)
        test_df['output'] = test_df['answer']
        test_df['input'] = test_df['question']
        if 'A' in test_df.columns and add_options:
            test_df['options'] = test_df.apply(lambda row: '\n'.join([f'{el}: ' + row[el] for el in 'ABCDE' if isinstance(row[el], str)]), axis=1)
            test_df['input'] = test_df.apply(lambda row: row['input'] + '\n' + row['options'], axis=1)
        else:
            test_df['input'] = test_df['question']
        test_df['instruction'] = ''
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if num_few_shot_samples > 0:
            few_shot_samples = test_df.iloc[:num_few_shot_samples]
            test_df = test_df.iloc[num_few_shot_samples:]
        else:
            few_shot_samples = None
        return test_df, few_shot_samples
