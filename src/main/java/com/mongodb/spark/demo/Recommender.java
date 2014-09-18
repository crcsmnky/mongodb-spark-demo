package com.mongodb.spark.demo;

import com.mongodb.hadoop.BSONFileInputFormat;
import com.mongodb.hadoop.MongoOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.mllib.recommendation.ALS;
import org.bson.BSONObject;
import org.bson.BasicBSONObject;
import org.slf4j.Logger;
import scala.Tuple2;

import java.util.*;


public class Recommender {

    private static String HDFS_HOST = "hdfs://crcsmnky.local:9000";
    private static String MONGODB_HOST = "mongodb://127.0.0.1:27017/";
//    private static int SCALE_MAX = 5;
//    private static int SCALE_MIN = 1;

//    public static Comparator<Rating> RatingComparator = new Comparator<Rating>() {
//        @Override
//        public int compare(Rating r1, Rating r2) {
//            return Double.valueOf(r1.rating() - r2.rating()).intValue();
//        }
//    };

    public static void main(String[] args) {
        if(args.length < 4) {
            System.err.println("Usage: Recommender <ratings.bson hdfs path> <users.bson hdfs path> <movies.bson hdfs path> <outputdb.collection>");
            System.err.println("Example: Recommender /movielens/ratings.bson /movielens/users.bson /movielens/movies.bson movielens.predictions");
            System.exit(-1);
        }

        String ratingsUri = HDFS_HOST + args[0];
        String usersUri =   HDFS_HOST + args[1];
        String moviesUri =  HDFS_HOST + args[2];
        String mongodbUri = MONGODB_HOST + args[3];

        SparkConf conf = new SparkConf().setAppName("SparkRecommender");
        JavaSparkContext sc = new JavaSparkContext(conf);
        Logger log = sc.sc().log();

        Configuration bsonDataConfig = new Configuration();
        bsonDataConfig.set("mongo.job.input.format", "com.mongodb.hadoop.BSONFileInputFormat");

        Configuration predictionsConfig = new Configuration();
        predictionsConfig.set("mongo.output.uri", mongodbUri);

        JavaPairRDD<Object,BSONObject> bsonRatingsData = sc.newAPIHadoopFile(
            ratingsUri, BSONFileInputFormat.class, Object.class,
                BSONObject.class, bsonDataConfig);

        JavaRDD<Rating> ratingsData = bsonRatingsData.map(
            new Function<Tuple2<Object,BSONObject>,Rating>() {
                public Rating call(Tuple2<Object,BSONObject> doc) throws Exception {
                    Integer userid = (Integer) doc._2.get("userid");
                    Integer movieid = (Integer) doc._2.get("movieid");
                    Number rating = (Number) doc._2.get("rating");
                    return new Rating(userid, movieid, rating.doubleValue());
                }
            }
        );

        log.warn("ratings = " + ratingsData.count());

        // keep this RDD in memory as much as possible, and spill to disk if needed
//        ratingsData.persist(StorageLevel.MEMORY_AND_DISK());

        JavaRDD<Object> userData = sc.newAPIHadoopFile(usersUri,
            BSONFileInputFormat.class, Object.class, BSONObject.class, bsonDataConfig).map(
            new Function<Tuple2<Object, BSONObject>, Object>() {
                @Override
                public Object call(Tuple2<Object, BSONObject> doc) throws Exception {
                    return doc._2.get("userid");
                }
            }
        );

        log.warn("users = " + userData.count());

        JavaRDD<Object> movieData = sc.newAPIHadoopFile(moviesUri,
            BSONFileInputFormat.class, Object.class, BSONObject.class, bsonDataConfig).map(
            new Function<Tuple2<Object, BSONObject>, Object>() {
                @Override
                public Object call(Tuple2<Object, BSONObject> doc) throws Exception {
                    return doc._2.get("movieid");
                }
            }
        );

        log.warn("movies = " + movieData.count());

        // generate complete pairing for all possible (user,movie) combinations
        JavaPairRDD<Object,Object> usersMovies = userData.cartesian(movieData);

        log.warn("usersMovies = " + usersMovies.count());

        // create the model from existing ratings data
        MatrixFactorizationModel model = ALS.train(ratingsData.rdd(), 10, 10, 0.01);

        // predict ratings
        JavaRDD<Rating> predictions = model.predict(usersMovies.rdd()).toJavaRDD();

//        // get the min/max ratings
//        final Rating minRating = predictions.min(RatingComparator);
//        final Rating maxRating = predictions.max(RatingComparator);
//
//        // normalize predicted ratings on a scale of SCALE_MIN to SCALE_MAX
//        JavaRDD<Rating> predictionsNormalized = predictions.map(
//            new Function<Rating, Rating>() {
//                @Override
//                public Rating call(Rating rating) throws Exception {
//                    double newRating = 1 + (rating.rating() - minRating.rating()) *
//                        (SCALE_MAX - SCALE_MIN) / (maxRating.rating() - minRating.rating());
//                    return new Rating(rating.user(), rating.product(), newRating);
//                }
//            }
//        );

        // create BSON RDD from normalized predictions
        JavaPairRDD<Object,BSONObject> predictionsOutput = predictions.mapToPair(
            new PairFunction<Rating, Object, BSONObject>() {
                @Override
                public Tuple2<Object, BSONObject> call(Rating rating) throws Exception {
                    BSONObject doc = new BasicBSONObject();
                    doc.put("userid", rating.user());
                    doc.put("movieid", rating.product());
                    doc.put("rating", rating.rating());
                    doc.put("timestamp", new Date());
                    // null key means an ObjectId will be generated on insert
                    return new Tuple2<Object, BSONObject>(null, doc);
                }
            }
        );

//        predictionsOutput.repartition(4);

        log.warn("writing " + predictionsOutput.count() + " documents to " + mongodbUri);

        predictionsOutput.saveAsNewAPIHadoopFile("file:///notapplicable",
            Object.class, Object.class, MongoOutputFormat.class, predictionsConfig);

        sc.sc().log().info("predictionsOutput.splits() = " + predictionsOutput.splits().size());
	}
}
