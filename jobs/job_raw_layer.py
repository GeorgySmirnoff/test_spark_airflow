import argparse
import os
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name
from utils.postgres import write_postgres_table


DATA_PATH = os.getenv("DATA_PATH", "/usr/local/airflow/data")


def sanitize_column_name(column_name: str) -> str:
    """Приводит имя колонки к безопасному формату PostgreSQL.

    Args:
        column_name: Исходное имя колонки из CSV.
    """

    name = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")

    if re.match(r"^\d", name):
        return f"d_{name}"

    return name


def prepare_column_names(df: DataFrame) -> DataFrame:
    """Переименовывает колонки DataFrame в безопасный формат.

    Args:
        df: Исходный DataFrame.
    """

    return df.toDF(*[sanitize_column_name(column_name) for column_name in df.columns])


def drop_empty_header_columns(df: DataFrame) -> DataFrame:
    """Удаляет колонки, созданные из пустых CSV-заголовков. Либо полностью пустые, либо с именами вида _c347

    Args:
        df: Исходный DataFrame.
    """

    columns_to_drop = [
        column_name
        for column_name in df.columns
        if not column_name.strip() or re.fullmatch(r"_c\d+", column_name)
    ]

    if columns_to_drop:
        print(f"Drop empty header columns: {columns_to_drop}")
        return df.drop(*columns_to_drop)

    return df


def read_raw_files(spark: SparkSession, data_path: str) -> DataFrame:
    """Читает и нормализует все RAW CSV-файлы.

    Args:
        spark: Активная Spark-сессия.
    """

    try:
        file_names = sorted(os.listdir(data_path))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Директория с файлами не найдена: {data_path}"
        ) from error

    files = [
        os.path.join(data_path, file_name)
        for file_name in file_names
        if file_name.startswith("raw_") and file_name.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError(
            f"В директории {data_path} не найдены raw_*.csv"
        )

    raw_df = None
    for file_number, file_path in enumerate(files, start=1):
        print(f"Обрабатывается файл {file_number}/{len(files)}: {file_path}")
        df = (
            spark.read.option("header", "true")
            .option("inferSchema", "false")
            .option("sep", ";")
            .csv(file_path)
        )

        df = drop_empty_header_columns(df)
        df = prepare_column_names(df)
        df = df.withColumn("source_file", input_file_name()).withColumn(
            "loaded_at",
            current_timestamp(),
        )

        if raw_df is None:
            raw_df = df
        else:
            raw_df = raw_df.unionByName(df, allowMissingColumns=True)

    return raw_df


def load_raw_layer(spark: SparkSession, raw_table: str, data_path: str) -> None:
    """Загружает CSV-файлы в RAW-слой.

    Args:
        spark: Активная Spark-сессия.
        raw_table: Целевая таблица RAW-слоя.
    """

    print("Start RAW layer loading")

    raw_df = read_raw_files(spark, data_path)

    print("RAW DataFrame schema:")
    raw_df.printSchema()

    print(f"RAW DataFrame rows count: {raw_df.count()}")

    write_postgres_table(raw_df, raw_table)
    print(f"RAW layer saved to {raw_table}")


def main() -> None:
    """Запускает джобу RAW-слоя."""
    parser = argparse.ArgumentParser(description="RAW layer job")
    parser.add_argument("--raw-table", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("job_raw_layer").getOrCreate()
    print("Spark session started")
    print(f"DATA_PATH = {DATA_PATH}")

    try:
        load_raw_layer(spark, args.raw_table, DATA_PATH)
    finally:
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
