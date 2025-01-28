import json
import argparse
import tiktoken
import numpy as np
from collections import defaultdict

def load_dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def initial_stats(dataset):
    print(f"Количество примеров: {len(dataset)}")
    if len(dataset) > 0:
        print("Первый пример:")
        for message in dataset[0].get("messages", []):
            print(message)
    else:
        print("Датасет пуст.")

def validate_format(dataset):
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1
    
    if format_errors:
        print("Найдены ошибки:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("Ошибок не найдено")

def num_tokens_from_messages(messages, encoding, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages, encoding):
    num_tokens = 0
    for message in messages:
        if message.get("role") == "assistant":
            num_tokens += len(encoding.encode(message.get("content", "")))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Распределение {name}:")
    print(f"мин / макс: {min(values)}, {max(values)}")
    print(f"среднее / медиана: {np.mean(values):.2f}, {np.median(values)}")
    print(f"p10 / p90: {np.quantile(values, 0.1):.2f}, {np.quantile(values, 0.9):.2f}")

def analyze_dataset(dataset):
    encoding = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS_PER_EXAMPLE = 16385

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []
    
    for ex in dataset:
        messages = ex.get("messages", [])
        if not any(message.get("role") == "system" for message in messages):
            n_missing_system += 1
        if not any(message.get("role") == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages, encoding))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages, encoding))
        
    print(f"\nКоличество примеров без системного сообщения: {n_missing_system}")
    print(f"Количество примеров без пользовательского сообщения: {n_missing_user}")
    print_distribution(n_messages, "количество сообщений на пример")
    print_distribution(convo_lens, "общее количество токенов на пример")
    print_distribution(assistant_message_lens, "количество токенов в сообщениях ассистента на пример")
    n_too_long = sum(l > MAX_TOKENS_PER_EXAMPLE for l in convo_lens)
    print(f"\n{n_too_long} примеров могут превышать лимит в {MAX_TOKENS_PER_EXAMPLE} токенов и будут усечены при дообучении")

def estimate_cost(dataset, epochs=3, min_target=100, max_target=25000, max_tokens=16385):
    MAX_TOKENS_PER_EXAMPLE = max_tokens
    TARGET_EPOCHS = epochs
    MIN_TARGET_EXAMPLES = min_target
    MAX_TARGET_EXAMPLES = max_target
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_train_examples = len(dataset)
    n_epochs = TARGET_EPOCHS
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    # Подсчёт токенов
    encoding = tiktoken.get_encoding("cl100k_base")
    convo_lens = [min(MAX_TOKENS_PER_EXAMPLE, num_tokens_from_messages(ex.get("messages", []), encoding)) for ex in dataset]
    n_billing_tokens_in_dataset = sum(convo_lens)
    
    print(f"\nДатасет содержит ~{n_billing_tokens_in_dataset} токенов, за которые будет начислено при обучении")
    print(f"По умолчанию будет обучено за {n_epochs} эпох(и) на этом датасете")
    print(f"По умолчанию будет начислено ~{n_epochs * n_billing_tokens_in_dataset} токенов")

def main():
    parser = argparse.ArgumentParser(description="Инструмент для подготовки данных к дообучению модели.")
    parser.add_argument("data_path", type=str, help="Путь к файлу с данными в формате JSONL")
    parser.add_argument("--epochs", type=int, default=3, help="Количество эпох для дообучения")
    parser.add_argument("--min_target", type=int, default=100, help="Минимальное количество примеров для дообучения")
    parser.add_argument("--max_target", type=int, default=25000, help="Максимальное количество примеров для дообучения")
    args = parser.parse_args()

    print("Загрузка данных...")
    dataset = load_dataset(args.data_path)
    initial_stats(dataset)
    
    print("\nПроверка формата данных...")
    validate_format(dataset)
    
    print("\nАнализ датасета...")
    analyze_dataset(dataset)
    
    print("\nОценка стоимости дообучения...")
    estimate_cost(dataset, epochs=args.epochs, min_target=args.min_target, max_target=args.max_target)

if __name__ == "__main__":
    main()
