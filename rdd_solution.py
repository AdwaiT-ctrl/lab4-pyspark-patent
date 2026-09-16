from pyspark import SparkContext, SparkConf
import csv
import io


def parse_csv_line(line):
    # robust CSV parsing for a single line
    return next(csv.reader(io.StringIO(line)))


def main():
    conf = SparkConf().setAppName("Lab4-RDD-Solution").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    rddC = sc.textFile('cite75_99.txt.gz')
    rddP = sc.textFile('apat63_99.txt.gz')

    # filter headers
    headerC = rddC.first()
    headerP = rddP.first()

    rddC_filtered = rddC.filter(lambda l: l != headerC).map(lambda l: parse_csv_line(l)).map(lambda fields: (int(fields[0]), int(fields[1])))

    def parse_patent(line):
        fields = parse_csv_line(line)
        try:
            pid = int(fields[0])
        except:
            return None
        state = fields[5] if len(fields) > 5 else ''
        return (pid, state)

    rddP_filtered = rddP.filter(lambda l: l != headerP).map(parse_patent).filter(lambda x: x is not None)

    # create KVs
    patents_kv = rddP_filtered

    # citations keyed by cited to join with patents to get cited state
    cited_keyed = rddC_filtered.map(lambda pair: (pair[1], pair[0]))  # (cited, citing)

    cited_with_state = cited_keyed.join(patents_kv)  # (cited, (citing, cited_state))

    # map to (citing, cited_state)
    citing_citedstate = cited_with_state.map(lambda x: (x[1][0], x[1][1]))

    # join to get citing state
    citing_with_states = citing_citedstate.join(patents_kv)  # (citing, (cited_state, citing_state))

    # filter where cited_state is not empty and equals citing_state
    same_state = citing_with_states.filter(lambda x: x[1][0] and x[1][0] == x[1][1])

    # count per citing patent
    same_state_counts = same_state.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)

    # get top 10
    top10 = same_state_counts.takeOrdered(10, key=lambda x: -x[1])

    print("Top 10 same-state citing patents (patent, count):")
    for p, c in top10:
        print(p, c)

    # augment original patents lines with count and write small CSV of augmented data
    patents_by_id = rddP.map(lambda l: parse_csv_line(l)).filter(lambda fields: fields[0] != 'PATENT')
    patents_by_id_kv = patents_by_id.map(lambda fields: (int(fields[0]), fields))

    augmented = patents_by_id_kv.leftOuterJoin(same_state_counts)  # (pid, (fields, count_opt))

    def format_augmented(x):
        pid, (fields, count) = x
        cnt = str(count) if count is not None else '0'
        # append count as last column
        return ",".join([str(f) for f in fields]) + "," + cnt

    augmented.map(format_augmented).coalesce(1).saveAsTextFile('patents_augmented_rdd')

    sc.stop()


if __name__ == '__main__':
    main()
