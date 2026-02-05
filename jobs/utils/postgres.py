import os

from pyspark.sql import DataFrame, SparkSession

POSTGRES_DRIVER = "org.postgresql.Driver"


def get_required_env(name: str) -> str:
    """Возвращает обязательную переменную окружения.

    Args:
        name: Имя переменной окружения.
    """

    try:
        return os.environ[name]
    except KeyError as error:
        raise RuntimeError(f"Не задана переменная окружения {name}") from error


def get_postgres_connection() -> dict:
    """Собирает параметры подключения к PostgreSQL."""

    host = get_required_env("POSTGRES_HOST")
    port = get_required_env("POSTGRES_PORT")
    database = get_required_env("POSTGRES_DB")
    user = get_required_env("POSTGRES_USER")
    password = get_required_env("POSTGRES_PASSWORD")

    return {
        "jdbc_url": f"jdbc:postgresql://{host}:{port}/{database}",
        "properties": {
            "user": user,
            "password": password,
            "driver": POSTGRES_DRIVER,
        },
    }


def read_postgres_table(spark: SparkSession, table_name: str) -> DataFrame:
    """Читает таблицу PostgreSQL через JDBC.

    Args:
        spark: Активная Spark-сессия.
        table_name: Исходная таблица со схемой.
    """

    connection = get_postgres_connection()
    return spark.read.jdbc(
        url=connection["jdbc_url"],
        table=table_name,
        properties=connection["properties"],
    )


def write_postgres_table(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
) -> None:
    """Записывает DataFrame в PostgreSQL.

    Args:
        df: DataFrame для записи.
        table_name: Целевая таблица со схемой.
        mode: Режим записи Spark.
    """

    connection = get_postgres_connection()
    df.write.mode(mode).jdbc(
        url=connection["jdbc_url"],
        table=table_name,
        properties=connection["properties"],
    )
