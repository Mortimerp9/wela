name := "Wela"

organization := "net.pierreandrews"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.0"

resolvers ++= Seq(
  "Sonatype OSS Releases" at "http://oss.sonatype.org/content/repositories/releases/",
  "Sonatype OSS Snapshots" at "http://oss.sonatype.org/content/repositories/snapshots/"
)


libraryDependencies ++= Seq(
  "org.specs2" %% "specs2" % "1.13" % "test",
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.6.9",
"org.scalaz" %% "scalaz-core" % "7.0.0",
"com.chuusai" %% "shapeless" % "1.2.5-SNAPSHOT"
)

initialCommands := "import wela._"
