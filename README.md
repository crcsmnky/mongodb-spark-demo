# MongoDB-Spark Demo

## Prerequisites

To build the MongoDB-Hadoop demo applications, you'll need to have the following:

* [Maven](http://maven.apache.org)
* [MongoDB-Hadoop Connector](http://github.com/mongodb/mongo-hadoop)
* [Spark](http://spark.apache.org) (1.0 or greater, for Hadoop 2.x)

### MongoDB-Hadoop

*Note*: you must build the MongoDB-Hadoop connector from source for your version of Hadoop and install the `core` JAR file to your local Maven repo. For example:

    $ git clone http://github.com/mongodb/mongo-hadoop.git
    $ cd mongo-hadoop
    $ ./gradlew jar -Phadoop_version='2.4'
    $ mvn install:install-file \
        -Dfile=core/build/lib/mongo-hadoop-core-1.2.1-SNAPSHOT-hadoop_2.4.jar \
        -DgroupId=com.mongodb \
        -DartifactId=hadoop \
        -Dversion=1.2.1-SNAPSHOT \
        -Dpackaging=jar

### Spark

Refer to the [Spark overview](http://spark.apache.org/docs/latest/index.html) to get started.

## Building

To the build the MongoDB-Hadoop demo applications use Maven:

    $ mvn package

This will build the demo application and place all of the dependencies in `target/lib`. If instead you want to build a single jar with all of the dependencies, execute the `assembly:single` Maven goal:

    $ mvn compile assembly:single

<!---
## Deploying

Deploy your JAR file to the appropriate location for your Hadoop distribution, e.g. `/usr/lib/hadoop/lib`.
--->

## Running

    $ cd your-spark-directory
    $ SPARK_JAR=assembly/target/scala-2.10/spark-assembly-1.0.0-hadoop2.4.0.jar \
      HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop \
      bin/spark-submit --master local --class com.mongodb.hadoop.demo.Recommender \
      ~/Work/Dropbox/Projects/mongodb-hadoop-demo/target/demo-1.0-SNAPSHOT.jar \
      --jars /path/to/mongo-java-driver-2.12.2.jar,/path/to/hadoop-1.2.1-SNAPSHOT.jar \
      --executor-memory 4G /movielens/ratings.bson /movielens/users.bson \
      /movielens/movies.bson movielens.predictions
        
## Notes

None at this time.
