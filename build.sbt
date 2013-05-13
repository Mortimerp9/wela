name := "Wela"

organization := "net.pierreandrews"

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.10.0"

libraryDependencies ++= Seq(
  "org.specs2" %% "specs2" % "1.13" % "test",
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.6.9",
"org.scalaz" %% "scalaz-core" % "7.0.0"
)

initialCommands := "import wela._"
