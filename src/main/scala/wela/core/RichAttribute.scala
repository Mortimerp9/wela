package wela.core

import weka.core.{ Attribute => WekaAttribute }
import weka.core.ProtectedProperties
import java.util.Properties


sealed trait Attribute {
  val name: Symbol
  type ValType <: AttributeValue
  protected type This <: Attribute
  def toWekaAttribute: WekaAttribute
  def changeName(newName: Symbol): This
}

case class NumericAttribute(override val name: Symbol,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends Attribute {
  override type ValType = NumericValue
  override lazy val toWekaAttribute = new WekaAttribute(name.name, metadata)
  override protected type This = NumericAttribute
  override def changeName(newName: Symbol) = copy(name=newName)
}

sealed trait NominalAttr extends Attribute {
  def levels: Seq[ValType]
}

sealed trait StringAttr extends NominalAttr {
  override type ValType = StringValue
}

case class StringAttribute(override val name: Symbol,
  override val levels: Seq[StringValue] = Nil,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override type ValType = StringValue
  override lazy val toWekaAttribute = new WekaAttribute(name.name, levels.to[FastVector], metadata)
  override protected type This = StringAttribute
  override def changeName(newName: Symbol) = copy(name=newName)
}

case class NominalAttribute(override val name: Symbol,
  override val levels: Seq[SymbolValue],
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override type ValType = SymbolValue
  override lazy val toWekaAttribute = new WekaAttribute(name.name, levels.map(_.name).to[FastVector], metadata)
  override protected type This = NominalAttribute
  override def changeName(newName: Symbol) = copy(name=newName)
}
