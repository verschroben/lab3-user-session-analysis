from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def task_1(df):
    print("\nЗадание 1. Ежедневная активность")

    result = (
        df.withColumn("date", F.to_date("event_time"))
        .groupBy("date")
        .agg(F.count("event_id").alias("total_events"))
        .orderBy("date")
    )

    result.show(100, truncate=False)


def task_2(df):
    print("\nЗадание 2. Анализ пользователей и сессий")

    print("\nУникальные пользователи")
    users = (
        df.select("user_id")
        .distinct()
        .orderBy("user_id")
    )
    users.show(100, truncate=False)

    print("\nОбщее количество уникальных сессий")
    total_sessions = (
        df.select("user_id", "session_id")
        .distinct()
        .count()
    )
    print(total_sessions)

    print("\nКоличество сессий каждого пользователя")
    sessions_per_user = (
        df.groupBy("user_id")
        .agg(
            F.countDistinct("session_id").alias(
                "unique_sessions_count"
            )
        )
        .orderBy("user_id")
    )
    sessions_per_user.show(100, truncate=False)


def task_3(df):
    print("\nЗадание 3. Детализация сессий")

    result = (
        df.groupBy("user_id", "session_id")
        .agg(
            F.count("event_id").alias(
                "count_events_in_session"
            )
        )
        .orderBy(
            "count_events_in_session",
            "user_id",
            "session_id"
        )
    )

    result.show(100, truncate=False)


def task_4(df):
    print("\nЗадание 4. Анализ покупок")

    purchases = df.filter(
        F.col("event_type") == "purchase"
    )

    result = purchases.agg(
        F.count("event_id").alias("count_purchase_amount"),
        F.sum("amount").alias("sum_purchase_amount"),
        F.round(F.avg("amount"), 2).alias(
            "average_purchase_amount"
        )
    )

    result.show(truncate=False)


def task_5(df):
    print("\nЗадание 5. Анализ продолжительности сессий")

    sessions = (
        df.groupBy("user_id", "session_id")
        .agg(
            F.min("event_time").alias("min_time"),
            F.max("event_time").alias("max_time")
        )
        .withColumn(
            "session_duration_seconds",
            F.unix_timestamp("max_time")
            - F.unix_timestamp("min_time")
        )
        .orderBy("user_id", "session_id")
    )

    print("\nПродолжительность каждой сессии")
    sessions.show(100, truncate=False)

    print("\nСредняя продолжительность сессий")
    average = sessions.agg(
        F.round(
            F.avg("session_duration_seconds"), 2
        ).alias("average_session_duration_seconds")
    )

    average.show(truncate=False)


def main():
    spark = (
        SparkSession.builder
        .appName("UserSessionsAnalysis")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-avro_2.12:3.5.9"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        file_path = (
            Path(__file__).resolve().parent / "activity_log.avro"
        )

        print("\nЗагрузка данных")
        df = spark.read.format("avro").load(str(file_path))

        print("\nПодготовка данных")
        df = df.withColumn(
            "event_time",
            F.to_timestamp("timestamp", "dd-MMM-yyyy HH:mm")
        )

        df.printSchema()

        print("\nПервые пять записей")
        df.show(5, truncate=False)

        print("\nПроверка преобразования времени")
        df.select(
            "timestamp", "event_time"
        ).show(5, truncate=False)

        if df.filter(F.col("event_time").isNull()).limit(1).count():
            raise ValueError(
                "Некоторые значения timestamp не преобразованы"
            )

        task_1(df)
        task_2(df)
        task_3(df)
        task_4(df)
        task_5(df)

        print("\nВсе задания выполнены")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
