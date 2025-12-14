import logging
from gigachat import GigaChat

import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GIGA_KEY = os.getenv("GIGA_KEY")

giga = GigaChat(
   credentials=GIGA_KEY,
   verify_ssl_certs=False
)

def giga_answer(query: str, fragments: list[dict]) -> str:
   logger.info(
       "Generating answer with GigaChat",
       extra={
           "query": query,
           "fragments_count": len(fragments)
       }
   )
   
   try:
       q = f"""
       Ваша роль - выступать в качестве системы информационного поиска.
       Вам будет задан вопрос, а также предоставлены релевантные отрывки из различных документов.
       Ваша задача - сформировать короткий и информативный ответ (не более 150 слов), основанный исключительно на представленных отрывках.
       Обязательно использовать информацию только из данных отрывков.
       Важно соблюдать нейтральный и объективный тон, а также избегать повторения текста.
       В конце формируйте окончательный ответ.
       Не пытайтесь изобрести ответ.
       Отвечайте исключительно на русском языке, за исключением специфических терминов.
       Если представленные документы не содержат информации, достаточной для формирования ответа, скажите: "Я не могу ответить на Ваш вопрос, используя информацию из предоставленной документации.Попробуйте переформулировать вопрос."
       Если документ содержит информацию, относящуюся к запросу, но запрос не предполагает прямого ответа, то просто перескажите содержание релевантного документа.
       Пиши в формате markdown.
       
       Вопрос пользователя: {query}
       
       Выдержки из документов:
       
       """

       for fragment in fragments:
          q += f"{fragment.text}\n"

       logger.debug("Sending request to GigaChat", extra={"prompt_length": len(q)})
       response = giga.chat(q)
       
       answer = response.choices[0].message.content
       logger.info(
           "Answer generated successfully",
           extra={
               "answer_length": len(answer) if answer else 0,
               "query": query
           }
       )
       
       return answer
   except Exception as e:
       logger.error(
           "Error generating answer with GigaChat",
           extra={
               "query": query,
               "fragments_count": len(fragments),
               "error": str(e)
           },
           exc_info=True
       )
       raise
