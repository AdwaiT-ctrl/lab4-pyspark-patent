from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce


def main():
    spark = SparkSession.builder.master("local[*]").appName("Lab4-DataFrame-Solution").getOrCreate()

    citations = spark.read.load('cite75_99.txt.gz', format='csv', sep=',', header=True, compression='gzip', inferSchema=True)
    patents = spark.read.load('apat63_99.txt.gz', format='csv', sep=',', header=True, compression='gzip', inferSchema=True)

    # normalize column names (uppercase) for robustness
    for df in (citations, patents):
        for c in df.columns:
            if c != c.upper():
                df = df.withColumnRenamed(c, c.upper())

    # reload to use normalized names (simple approach: create new references)
    citations = spark.read.load('cite75_99.txt.gz', format='csv', sep=',', header=True, compression='gzip', inferSchema=True)
    patents = spark.read.load('apat63_99.txt.gz', format='csv', sep=',', header=True, compression='gzip', inferSchema=True)

    # Select only needed columns and rename for clarity
    patents_min = patents.select(col('PATENT').alias('PATENT_ID'), col('POSTATE').alias('STATE'))
    citations_min = citations.select(col('CITING').alias('CITING_ID'), col('CITED').alias('CITED_ID'))

    # Join to get cited patent state
    cited_join = citations_min.join(patents_min, citations_min.CITED_ID == patents_min.PATENT_ID, how='left') \
        .select('CITING_ID', 'CITED_ID', col('STATE').alias('CITED_STATE'))

    # Join to get citing patent state
    cited_and_citing = cited_join.join(patents_min, cited_join.CITING_ID == patents_min.PATENT_ID, how='left') \
        .select('CITING_ID', 'CITED_ID', 'CITED_STATE', col('STATE').alias('CITING_STATE'))

    # Count same-state citations (only when cited state is not null)
    same_state = cited_and_citing.filter((col('CITED_STATE').isNotNull()) & (col('CITED_STATE') == col('CITING_STATE'))) \
        .groupBy('CITING_ID').count().withColumnRenamed('count', 'SAME_STATE_CITATIONS')

    # Augment patents with same-state counts
    patents_aug = patents.join(same_state, patents.PATENT == same_state.CITING_ID, how='left') \
        .withColumn('SAME_STATE_CITATIONS', coalesce(col('SAME_STATE_CITATIONS'), col('SAME_STATE_CITATIONS')*0 + 0))

    # Show top 10 patents by same-state citations
    top10 = patents_aug.select('PATENT', 'POSTATE', 'SAME_STATE_CITATIONS') \
        .orderBy(col('SAME_STATE_CITATIONS').desc())

    print("Top 10 patents by same-state citations:")
    top10.show(10, False)

    # write only the top-10 results to CSV (limit before writing)
    top10.limit(10).coalesce(1).write.csv('top10_same_state_dataframe.csv', header=True, mode='overwrite')

    spark.stop()


if __name__ == '__main__':
    main()
