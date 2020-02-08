Consensus DL.



## Installing the code on Linux

Delete the contents of the target folder and run the following commands:

1. mvn eclipse:eclipse

2. mvn install:install-file -Dfile=lib/djep-1.0.0.jar -DgroupId=org.lsmp -DartifactId=djep -Dversion=1.0.0 -Dpackaging=jar

3. mvn install:install-file -Dfile=lib/jep-2.3.0.jar -DgroupId=org.nfunk -DartifactId=jep -Dversion=2.3.0 -Dpackaging=jar

4. mvn compile
