import argparse
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    expr,
    regexp_replace,
    to_date,
    trim,
    when,
)
from utils.postgres import read_postgres_table, write_postgres_table


DATE_COLUMN_PATTERN = re.compile(r"^d_\d{1,2}_\d{1,2}_\d{2,4}$")


def collect_date_columns(df: DataFrame) -> list[str]:
    """Находит очищенные колонки с датами в RAW DataFrame.

    Args:
        df: DataFrame RAW-слоя.
    """

    date_columns = [
        column_name
        for column_name in df.columns
        if DATE_COLUMN_PATTERN.fullmatch(column_name)
    ]

    if not date_columns:
        raise ValueError("В RAW-таблице не найдены колонки с датами")

    return date_columns


def build_stack_expression(date_columns: list[str]) -> str:
    """Собирает Spark stack-выражение для разворота дат.

        stack(
            3,
            'd_1_22_20', `d_1_22_20`,
            'd_1_23_20', `d_1_23_20`,
            'd_1_24_20', `d_1_24_20`
        ) as (date_key, cumulative_deaths_raw)

    Args:
        date_columns: Колонки с датами для разворота.
    """

    stack_args = ", ".join(
        f"'{column_name}', `{column_name}`" for column_name in date_columns
    )
    return (
        f"stack({len(date_columns)}, {stack_args}) "
        "as (date_key, cumulative_deaths_raw)"
    )


def build_stage_dataframe(raw_df: DataFrame) -> DataFrame:
    """Преобразует RAW-данные в структуру STAGE-слоя.

    Args:
        raw_df: DataFrame RAW-слоя.
    """

    date_columns = collect_date_columns(raw_df)
    print(f"Found date columns count: {len(date_columns)}")

    return (
        raw_df.select(
            col("country_region"),
            col("province_state"),
            col("source_file"),
            # Разворачиваем date-колонки RAW в пары:
            # date_key = имя исходной колонки, cumulative_deaths_raw = значение.
            expr(build_stack_expression(date_columns)),
        )
        # Убираем технический префикс d_ из имени date-колонки.
        .withColumn("date_key", regexp_replace(col("date_key"), "^d_", ""))
        # Парсим date_key в тип date: поддерживаем форматы M_d_yy и M_d_yyyy.
        .withColumn(
            "report_date",
            when(
                col("date_key").rlike(r"^\d{1,2}_\d{1,2}_\d{2}$"),
                to_date(col("date_key"), "M_d_yy"),
            ).when(
                col("date_key").rlike(r"^\d{1,2}_\d{1,2}_\d{4}$"),
                to_date(col("date_key"), "M_d_yyyy"),
            ),
        )
        .withColumn(
            "cumulative_deaths",
            # Убираем пробелы и приводим накопленное значение смертей к bigint.
            trim(col("cumulative_deaths_raw")).cast("bigint"),
        )
        # Оставляем только строки с успешно распарсенной датой.
        .filter(col("report_date").isNotNull())
        # Оставляем только строки с числовым значением смертей.
        .filter(col("cumulative_deaths").isNotNull())
        .select(
            col("country_region"),
            col("province_state"),
            col("report_date"),
            col("cumulative_deaths"),
            col("source_file"),
            current_timestamp().alias("processed_at"),
        )
    )


def build_stage_layer(
    spark: SparkSession,
    raw_table: str,
    stage_table: str,
) -> None:
    """Собирает и сохраняет STAGE-слой.

    Args:
        spark: Активная Spark-сессия.
        raw_table: Исходная таблица RAW-слоя.
        stage_table: Целевая таблица STAGE-слоя.
    """

    print("Read raw table")
    raw_df = read_postgres_table(spark, raw_table)

    print("Build stage table")
    stage_df = build_stage_dataframe(raw_df)

    print("STAGE DataFrame sample:")
    stage_df.show(10, truncate=False)

    print("Write stage table")
    write_postgres_table(stage_df, stage_table)
    print(f"STAGE layer saved to {stage_table}")


def main() -> None:
    """Запускает джобу STAGE-слоя."""
    parser = argparse.ArgumentParser(description="STAGE layer job")
    parser.add_argument("--raw-table", required=True)
    parser.add_argument("--stage-table", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("job_stage_layer").getOrCreate()
    print("Spark session started")

    try:
        build_stage_layer(spark, args.raw_table, args.stage_table)
    finally:
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
