from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Union
from pymilvus import connections, db, utility, FieldSchema, DataType, Collection, CollectionSchema

Vector = Union[List[float], Sequence[float]]


class MilvusSingleton:
    _instance: Optional["MilvusSingleton"] = None
    _lock: Lock = Lock()

    def __new__(cls, *args, **kwargs):
        # Потокобезопасный singleton
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, host: str = "localhost", port: str = "19530", alias: str = "default"):
        # __init__ будет вызываться при каждом MilvusSingleton(), защищаемся
        if getattr(self, "_initialized", False):
            return

        self.host = host
        self.port = port
        self.alias = alias

        self._initialize_connection()
        self._initialized = True

    def _initialize_connection(self):
        try:
            if connections.has_connection(self.alias):
                print(f"[INFO]: Milvus already connected (alias='{self.alias}')")
                return

            connections.connect(alias=self.alias, host=self.host, port=self.port)
            print(f"[INFO]: Connected to Milvus (alias='{self.alias}', {self.host}:{self.port})")
        except Exception as e:
            print(f"[ERROR]: Failed to connect to Milvus: {e}")
            raise

    ############################################################## Подключение к БД
    ## Настраивает базу данных с указанным именем
    def setup_database(self, db_name: str):
        existing = db.list_database(using=self.alias)
        if db_name not in existing:
            db.create_database(db_name=db_name, using=self.alias)
        db.using_database(db_name, using=self.alias)
        print(f"[INFO]: Using database '{db_name}'")

    ############################################################## Насчтройка схемы
    ## Создадим схему коллекции
    def create_schema(self, size_vec: int) -> CollectionSchema:
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
        source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255)
        embedding_field = FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=size_vec)
        content_field = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192)

        return CollectionSchema(fields=[id_field, source_field, embedding_field, content_field])

    ## Удаление коллекции
    def delete_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"[INFO]: Collection '{collection_name}' existed and was deleted.")
        else:
            print(f"[INFO]: Collection '{collection_name}' does not exist.")

    ## Получение коллекции по имени
    def get_collection(self, collection_name: str) -> Collection:
        return Collection(name=collection_name)

    ## Создадим коллекцию в БД
    def create_collection(self, collection_name: str, size_vec: int, drop_if_exists: bool = False):
        if utility.has_collection(collection_name):
            if drop_if_exists:
                self.delete_collection(collection_name)
            else:
                print(f"[INFO]: Collection '{collection_name}' already exists.")
                # на всякий случай загрузим/проверим индекс
                self.create_index_load(collection_name)
                return

        schema = self.create_schema(size_vec)
        Collection(name=collection_name, schema=schema)
        print(f"[INFO]: Create collection '{collection_name}'")
        self.create_index_load(collection_name)

    ############################################################## Настройка индекса поиска и загрузка данных
    ## Прописываем нужные параметры индекса
    def create_index_params(self):
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 5},
        }

    def create_search_params(self) -> Dict[str, Any]:
        return {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 5},
        }

    ## Загружаем наш индекс поиска в коллекцию
    def create_index_load(self, collection_name: str):
        collection = self.get_collection(collection_name)
        try:
            has_any_index = bool(getattr(collection, "indexes", []))
        except Exception:
            has_any_index = False

        if not has_any_index:
            index_params = self.create_index_params()
            collection.create_index(field_name="embeddings", index_params=index_params)
            print(f"[INFO]: Create index in '{collection_name}'")
        else:
            print(f"[INFO]: Index already exists in '{collection_name}'")

        collection.load()
        print(f"[INFO]: Collection '{collection_name}' loaded")

    ## Вставка данных в коллекцию
    def insert_data(
            self,
            collection_name: str,
            data: Dict[str, Any],
            flush: bool = False,
    ):
        required = ("id", "source", "embeddings", "content")
        for k in required:
            if k not in data:
                raise ValueError(f"data must contain key '{k}'")

        ids = data["id"]
        sources = data["source"]
        embeddings = data["embeddings"]
        contents = data["content"]

        collection = self.get_collection(collection_name)
        collection.insert([ids, sources, embeddings, contents])
        if flush:
            collection.flush()
        print(f"[INFO]: Inserted {len(ids)} rows into '{collection_name}'")

    ############################################################## Поиск по коллекции
    ## Поиск данных в коллекции
    def search_by_vector(
            self,
            query_embedding: Vector,
            collection_name: str,
            limit: int = 15,
    ) -> Dict[str, Any]:
        collection = self.get_collection(collection_name)
        collection.load()

        results = collection.search(
            data=[query_embedding],
            anns_field="embeddings",
            param=self.create_search_params(),
            limit=limit,
            output_fields=["source", "content"],
        )
        return self.filter_results(results)

    ## Обработка результата
    def filter_results(self, results) -> Dict[str, Any]:
        data = {"id": [], "distance": [], "source": [], "content": []}
        for hit in results[0]:
            data["id"].append(hit.id)
            data["distance"].append(hit.distance)
            data["source"].append(hit.entity.get("source"))
            data["content"].append(hit.entity.get("content"))
        return data