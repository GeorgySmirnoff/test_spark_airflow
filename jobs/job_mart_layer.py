import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    current_timestamp,
    lag,
    month,
    row_number,
    sum as spark_sum,
    when,
    year,
)
from pyspark.sql.window import Window
from utils.postgres import read_postgres_table, write_postgres_table


def build_monthly_cumulative(stage_df: DataFrame) -> DataFrame:
    """Считает накопленные смерти на конец каждого месяца.

    Args:
        stage_df: DataFrame ежедневных смертей из STAGE-слоя.
    """

    month_end_window = Window.partitionBy(
        "country_region",
        "province_state",
        "report_year",
        "report_month",
    ).orderBy(col("report_date").desc())

    return (
        stage_df
        # Выделяем год и месяц из report_date.
        .withColumn("report_year", year(col("report_date")))
        .withColumn("report_month", month(col("report_date")))
        # Нумеруем даты внутри страны/провинции и месяца от самой поздней к ранней.
        .withColumn("row_number", row_number().over(month_end_window))
        # Оставляем последнюю доступную дату месяца для каждой страны/провинции.
        .filter(col("row_number") == 1)
        # Группируем month-end записи по году и месяцу.
        .groupBy("report_year", "report_month")
        # Суммируем накопленные значения по всем странам и провинциям.
        .agg(spark_sum("cumulative_deaths").alias("cumulative_deaths_month_end"))
        .orderBy("report_year", "report_month")
    )


def build_mart_dataframe(stage_df: DataFrame) -> DataFrame:
    """Преобразует ежедневные STAGE-данные в структуру MART-слоя.

    Args:
        stage_df: DataFrame ежедневных смертей из STAGE-слоя.
    """

    # Окно задает хронологический порядок месяцев для расчета lag.
    month_window = Window.orderBy("report_year", "report_month")
    monthly_df = build_monthly_cumulative(stage_df)

    return (
        monthly_df
        # Добавляем накопленное значение на конец предыдущего месяца.
        .withColumn(
            "previous_cumulative_deaths_month_end",
            lag("cumulative_deaths_month_end").over(month_window),
        )
        # Считаем смерти за месяц как разницу между текущим и предыдущим накопленным значением.
        .withColumn(
            "monthly_deaths", 
            when(
                col("previous_cumulative_deaths_month_end").isNull(),
                col("cumulative_deaths_month_end"),
            ).otherwise(
                col("cumulative_deaths_month_end")
                - col("previous_cumulative_deaths_month_end")
            ),
        )
        # Оставляем только поля финальной MART-витрины.
        .select(
            col("report_year"),
            col("report_month"),
            col("cumulative_deaths_month_end"),
            col("monthly_deaths"),
            current_timestamp().alias("processed_at"),
        )
        .orderBy("report_year", "report_month")
    )


def build_mart_layer(
    spark: SparkSession,
    stage_table: str,
    mart_table: str,
) -> None:
    """Собирает и сохраняет MART-слой.

    Args:
        spark: Активная Spark-сессия.
        stage_table: Исходная таблица STAGE-слоя.
        mart_table: Целевая таблица MART-слоя.
    """

    print("Start MART layer building")
    stage_df = read_postgres_table(spark, stage_table)

    print("STAGE table schema from PostgreSQL:")
    stage_df.printSchema()

    mart_df = build_mart_dataframe(stage_df)

    print("MART DataFrame schema:")
    mart_df.printSchema()

    print("MART DataFrame sample:")
    mart_df.show(20, truncate=False)

    write_postgres_table(mart_df, mart_table)
    print(f"MART layer saved to {mart_table}")


def main() -> None:
    """Запускает джобу MART-слоя."""
    parser = argparse.ArgumentParser(description="MART layer job")
    parser.add_argument("--stage-table", required=True)
    parser.add_argument("--mart-table", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("job_mart_layer").getOrCreate()
    print("Spark session started")

    try:
        build_mart_layer(spark, args.stage_table, args.mart_table)
    finally:
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
