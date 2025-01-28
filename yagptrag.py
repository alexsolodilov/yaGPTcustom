import requests
import json
import os
import sys
from jsonschema import validate, ValidationError
import re
import string
from collections import Counter

def get_iam_token(oauth_token):
    """
    Обменивает OAuth-токен на IAM-токен.
    
    :param oauth_token: OAuth-токен (строка)
    :return: IAM-токен (строка)
    :raises: Exception, если обмен не удался
    """
    url = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    headers = {
        "Content-Type": "application/json"
    }
    body = {
        "yandexPassportOauthToken": oauth_token
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
        response.raise_for_status()
        iam_token = response.json().get("iamToken")
        if not iam_token:
            raise Exception("Не удалось получить IAM-токен из ответа.")
        return iam_token
    except requests.HTTPError as http_err:
        raise Exception(f"HTTP ошибка при получении IAM-токена: {http_err} - {response.text}") from http_err
    except Exception as err:
        raise Exception(f"Ошибка при получении IAM-токена: {err}") from err

def validate_json(instance, schema):
    """
    Валидирует JSON по заданной схеме.
    
    :param instance: JSON-данные для проверки
    :param schema: Схема для валидации
    :raises: ValidationError, если валидация не прошла
    """
    validate(instance=instance, schema=schema)

def generate_text_completion(iam_token, model_uri, messages, temperature=0.3, max_tokens=300, stream=False):
    """
    Генерирует текстовое продолжение с использованием Yandex Cloud Foundation Models API.

    :param iam_token: IAM-токен для авторизации
    :param model_uri: ID модели (например, 'b1g9no86lklqacng5kr3')
    :param messages: Список сообщений для контекста генерации
    :param temperature: Параметр креативности (0.0 - 1.0)
    :param max_tokens: Максимальное количество токенов в ответе
    :param stream: Включение потоковой передачи (True/False)
    :return: Ответ API в формате JSON
    :raises: Exception, если запрос не удался
    """
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json"
    }
    body = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": stream,
            "temperature": temperature,
            "maxTokens": str(max_tokens)  # Согласно документации, должно быть строкой
        },
        "messages": messages,
        "tools": []  # Пока не поддерживается, но включено для соответствия структуре
    }

    # Определение схемы JSON для валидации
    schema = {
        "type": "object",
        "properties": {
            "modelUri": {"type": "string"},
            "completionOptions": {
                "type": "object",
                "properties": {
                    "stream": {"type": "boolean"},
                    "temperature": {"type": "number"},
                    "maxTokens": {"type": "string"}
                },
                "required": ["stream", "temperature", "maxTokens"]
            },
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "text": {"type": "string"},
                        "toolCallList": {"type": "object"},
                        "toolResultList": {"type": "object"}
                    },
                    "required": ["role"]
                }
            },
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "function": {"type": "object"}
                    },
                    "required": ["function"]
                }
            }
        },
        "required": ["modelUri", "completionOptions", "messages"]
    }

    # Валидация JSON перед отправкой
    try:
        validate_json(body, schema)
        print("JSON прошёл валидацию.")
    except ValidationError as e:
        raise Exception(f"JSON не прошёл валидацию: {e.message}") from e

    # Вывод отправляемого запроса для отладки
    print("\n=== Отправляемый запрос ===")
    print(json.dumps(body, ensure_ascii=False, indent=4))
    print("===========================\n")

    try:
        response = requests.post(url, headers=headers, data=json.dumps(body))
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        try:
            error_info = response.json()
        except ValueError:
            error_info = response.text
        raise Exception(f"HTTP ошибка при генерации текста: {http_err} - {error_info}") from http_err
    except Exception as err:
        raise Exception(f"Ошибка при генерации текста: {err}") from err

def load_documentation(file_path):
    """
    Загружает документацию из JSON файла.
    
    :param file_path: Путь к файлу документации
    :return: Список разделов документации
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documentation = json.load(f)
        if isinstance(documentation, dict):
            documentation = [documentation]
        elif not isinstance(documentation, list):
            raise Exception("Неподдерживаемый формат документации. Ожидается список или объект.")
        return documentation
    except Exception as e:
        raise Exception(f"Не удалось загрузить документацию: {e}") from e

def extract_keywords(text, num_keywords=5):
    """
    Извлекает ключевые слова из текста запроса.
    
    :param text: Текст для извлечения ключевых слов
    :param num_keywords: Количество ключевых слов для извлечения
    :return: Список ключевых слов
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r'\b\w+\b', text)
    stop_words = set([
        'и', 'в', 'на', 'с', 'для', 'что', 'как', 'по', 'это', 'от', 'к', 
        'но', 'так', 'также', 'такой', 'быть', 'бы', 'все', 'во', 'его', 
        'её', 'еще', 'же', 'за', 'из', 'или', 'им', 'их', 'ихний', 'который', 
        'меня', 'меняю', 'мы', 'наш', 'не', 'него', 'негося', 'нет', 'ни', 
        'них', 'ну', 'о', 'об', 'один', 'она', 'они', 'оно', 'откуда', 
        'почему', 'поэтому', 'при', 'про', 'раз', 'сам', 'свой', 'себя', 
        'со', 'там', 'то', 'тоже', 'только', 'тот', 'у', 'уж', 'чтобы', 
        'через', 'чем', 'что-то', 'эта', 'эти', 'это', 'я'
    ])
    words = [word for word in words if word not in stop_words]
    word_counts = Counter(words)
    most_common = word_counts.most_common(num_keywords)
    keywords = [word for word, count in most_common]
    return keywords

def find_relevant_sections(documentation, keywords, top_n=3):
    """
    Находит релевантные разделы документации на основе ключевых слов.
    
    :param documentation: Список разделов документации
    :param keywords: Список ключевых слов
    :param top_n: Количество топ релевантных разделов для возврата
    :return: Список релевантных разделов
    """
    relevance = []
    for section in documentation:
        text = f"{section.get('anchor_text', '')} {section.get('section_text', '')}".lower()
        matches = sum(1 for keyword in keywords if keyword in text)
        relevance.append((matches, section))
    # Сортировка по количеству совпадений
    relevance.sort(key=lambda x: x[0], reverse=True)
    # Выбор топ-N
    top_sections = [section for match, section in relevance if match > 0][:top_n]
    return top_sections

def append_relevant_documentation_to_system(relevant_sections, messages, max_total_chars=32768):
    """
    Добавляет релевантные разделы документации в системное сообщение.
    
    :param relevant_sections: Список релевантных разделов документации
    :param messages: Список сообщений
    :param max_total_chars: Максимальное количество символов для системного сообщения
    """
    if not relevant_sections:
        return
    documentation_texts = [f"{section.get('anchor_text', '')}: {section.get('section_text', '')}" for section in relevant_sections]
    combined_documentation = "\n\n".join(documentation_texts)
    
    # Ограничение на количество символов
    if len(combined_documentation) > max_total_chars:
        combined_documentation = combined_documentation[:max_total_chars] + "..."
    
    # Находим системное сообщение
    system_message = next((msg for msg in messages if msg['role'] == 'system'), None)
    if system_message:
        # Добавляем релевантную документацию к существующему тексту
        system_text = system_message['text']
        system_text += f"\n\nРелевантная документация:\n{combined_documentation}"
        # Ограничиваем общий размер текста
        system_message['text'] = trim_text_to_limit(system_text, max_total_chars)
    else:
        # Если системного сообщения нет, создаем новое
        documentation_message = {
            "role": "system",
            "text": f"Релевантная документация:\n{combined_documentation}"
        }
        messages.append(documentation_message)

def append_user_message(user_input, assistant_response, messages, max_total_chars=32768):
    """
    Добавляет пользовательский запрос и ответ ассистента в один блок 'user'.
    
    :param user_input: Текущий запрос пользователя
    :param assistant_response: Ответ ассистента
    :param messages: Список сообщений
    :param max_total_chars: Максимальное количество символов для пользовательского сообщения
    """
    user_message = next((msg for msg in messages if msg['role'] == 'user'), None)
    if user_message:
        # Форматируем новый ввод и ответ
        formatted_text = f"{user_input}\n- {assistant_response}"
        # Добавляем к существующему тексту
        user_text = user_message['text'] + "\n" + formatted_text
        # Ограничиваем общий размер текста
        user_message['text'] = trim_text_to_limit(user_text, max_total_chars)
    else:
        # Если пользовательского сообщения нет, создаем новое
        combined_text = f"{user_input}\n- {assistant_response}"
        user_message = {
            "role": "user",
            "text": combined_text
        }
        messages.append(user_message)

def trim_text_to_limit(text, max_chars):
    """
    Обрезает текст до указанного количества символов.
    
    :param text: Текст для обрезки
    :param max_chars: Максимальное количество символов
    :return: Обрезанный текст
    """
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def main():
    """
    Основная функция скрипта.
    """
    # Инициализация списка сообщений с системным сообщением
    messages = [
        {
            "role": "system",
            "text": (
                "Ты эксперт сервиса Яндекс Маршрутизация.\n"
                "Ты отвечаешь на вопросы пользователей, которые они присылают в техническую поддержку."
            )
        }
    ]
    
    # Загрузка документации
    documentation_file = 'vrp.json'  # Укажите путь к вашему файлу документации
    try:
        documentation = load_documentation(documentation_file)
        print("Документация успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке документации: {e}")
        sys.exit(1)
    
    # Получение OAuth-токена из переменной окружения или аргумента командной строки
    oauth_token = ''  # Лучше использовать переменные окружения для безопасности
    if not oauth_token:
        if len(sys.argv) > 1:
            oauth_token = sys.argv[1]
            print("Используется OAuth-токен, переданный как аргумент командной строки.")
        else:
            oauth_token = input("Введите ваш OAuth-токен: ").strip()
            if not oauth_token:
                print("OAuth-токен не был предоставлен. Завершение работы.")
                sys.exit(1)

    try:
        # Обмен OAuth-токена на IAM-токен
        print("Получение IAM-токена...")
        iam_token = get_iam_token(oauth_token)
        print("IAM-токен успешно получен.")
    except Exception as e:
        print(f"Ошибка при получении IAM-токена: {e}")
        sys.exit(1)

    # Параметры запроса
    model_id = "gpt://b1g9no86lklqacng5kr3/yandexgpt-lite/latest@tamrc5bl63g7ss9hhkmpk"  # ID вашей модели

    while True:
        user_input = input("\nВведите ваш запрос к модели (или 'exit' для выхода): ").strip()
        if user_input.lower() == 'exit':
            print("Завершение работы.")
            break

        if not user_input:
            print("Пустой ввод. Пожалуйста, введите запрос или 'exit' для выхода.")
            continue

        # Извлечение ключевых слов из запроса
        keywords = extract_keywords(user_input)
        print(f"Извлеченные ключевые слова: {keywords}")

        # Поиск релевантных разделов документации
        relevant_sections = find_relevant_sections(documentation, keywords, top_n=3)
        if relevant_sections:
            print(f"Найдены релевантные разделы документации:")
            for section in relevant_sections:
                print(f"- {section.get('anchor_text', '')}")
            # Добавление релевантных разделов в системное сообщение
            append_relevant_documentation_to_system(relevant_sections, messages)
        else:
            print("Релевантные разделы документации не найдены.")

        # Создание временного списка сообщений для запроса
        temp_messages = [
            msg for msg in messages  # Копируем существующие сообщения
        ]

        # Добавление пользовательского запроса в temp_messages
        temp_messages.append({
            "role": "user",
            "text": user_input
        })

        # Подсчёт общего количества символов в запросе
        total_chars = sum(len(message.get('text', '')) for message in temp_messages)
        if total_chars > 32768:  # 8192 токенов ≈ 32768 символов
            print("Превышен лимит токенов. Пожалуйста, сократите ваш запрос или предыдущую переписку.")
            continue

        # Параметры генерации
        temperature = 0.3  # Параметр креативности
        max_tokens = 300    # Максимальное количество токенов в ответе
        stream = False      # Потоковая передача (не поддерживается в текущей реализации)

        try:
            # Генерация текста
            print("Отправка запроса на генерацию текста...")
            result = generate_text_completion(
                iam_token=iam_token,
                model_uri=model_id,
                messages=temp_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            print("Запрос успешно выполнен.\n")
        except Exception as e:
            print(f"Ошибка при генерации текста: {e}")
            continue  # Переход к следующему запросу

        # Обработка и вывод ответа
        try:
            # Получение альтернатив
            result_content = result.get('result', {})
            alternatives = result_content.get('alternatives', [])

            if not alternatives:
                print("Нет альтернатив для отображения.")
                continue

            # Предполагаем, что берем первую альтернативу
            alternative = alternatives[0]
            message = alternative.get('message', {})
            role = message.get('role', 'unknown')
            text = message.get('text', '')
            tool_calls = message.get('toolCallList', {}).get('toolCalls', [])
            tool_results = message.get('toolResultList', {}).get('toolResults', [])

            print(f"--- Ответ модели ---")
            if text:
                print(f"{role}: {text}\n")
            if tool_calls:
                print(f"{role} вызвал(и) инструменты:")
                for tool_call in tool_calls:
                    function_call = tool_call.get('functionCall', {})
                    name = function_call.get('name', 'unknown')
                    arguments = tool_call.get('arguments', {})
                    print(f"  Функция: {name}, Аргументы: {json.dumps(arguments, ensure_ascii=False)}")
                print()
            if tool_results:
                print(f"{role} получил(и) результаты инструментов:")
                for tool_result in tool_results:
                    function_result = tool_result.get('functionResult', {})
                    name = function_result.get('name', 'unknown')
                    content = function_result.get('content', '')
                    print(f"  Функция: {name}, Результат: {content}")
                print()

            # Добавление ответа ассистента к пользовательскому сообщению
            if text:
                append_user_message(user_input, text, messages)

            # Вывод статистики использования
            usage = result.get('usage', {})
            print("--- Статистика использования ---")
            print(f"Входящие токены: {usage.get('inputTextTokens', 'N/A')}")
            print(f"Токены генерации: {usage.get('completionTokens', 'N/A')}")
            print(f"Всего токенов: {usage.get('totalTokens', 'N/A')}")

            # Вывод версии модели
            model_version = result.get('modelVersion', 'N/A')
            print(f"\nВерсия модели: {model_version}")

        except Exception as e:
            print(f"Ошибка при обработке ответа: {e}")

if __name__ == "__main__":
    main()
