package wela.core

import weka.core.{ Attribute => WekaAttribute }
import weka.core.ProtectedProperties
import java.util.Properties

sealed trait Attribute {
  val name: Symbol
  def toWekaAttribute: WekaAttribute
}

sealed trait NumericAttr extends Attribute

case class NumericAttribute(override val name: Symbol,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NumericAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, metadata)
}
case class IndexedAttribute(override val name: Symbol, index: Int) extends NumericAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, index)
}

sealed trait NominalAttr extends Attribute {
  def values: Seq[Symbol]
}

case class StringAttribute(override val name: Symbol,
  override val values: Seq[Symbol] = Nil,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, values.map(_.name).to[FastVector], metadata)
}
case class IndexedStringAttribute(override val name: Symbol, index: Int, override val values: Seq[Symbol] = Nil) extends NominalAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, values.map(_.name).to[FastVector], index)
}

case class NominalAttribute(override val name: Symbol,
  override val values: Seq[Symbol],
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, values.map(_.name).to[FastVector], metadata)
}
case class IndexedNominalAttribute(override val name: Symbol, index: Int, override val values: Seq[Symbol]) extends NominalAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, values.map(_.name).to[FastVector], index)
}

sealed trait DateAttr extends Attribute {
  def dateFormat: String
}

case class DateAttribute(override val name: Symbol, override val dateFormat: String,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends DateAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, dateFormat, metadata)
}
case class IndexedDateAttribute(override val name: Symbol, index: Int, override val dateFormat: String) extends DateAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, dateFormat, index)
}

sealed trait AttributeValue {
  type T
  def value: T
}
case class StringValue(override val value: String) extends AttributeValue {
  override type T = String
}
case class DoubleValue(override val value: Double) extends AttributeValue {
  override type T = Double
}
