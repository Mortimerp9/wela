package wela.core

import weka.core.{ Attribute => WekaAttribute }
import weka.core.ProtectedProperties
import java.util.Properties

object Attribute {

  def apply(name: Symbol, value: AttributeValue): Attribute = value match {
    case v: StringValue => StringAttribute(name)
    case v: SymbolValue => NominalAttribute(name, Seq())
    case v: NumericValue => NumericAttribute(name)
  }
  
}

sealed trait Attribute {
  val name: Symbol
  type ValType <: AttributeValue
  def toWekaAttribute: WekaAttribute
}

sealed trait NumericAttr extends Attribute {
  override type ValType = NumericValue
}

case class NumericAttribute(override val name: Symbol,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NumericAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, metadata)
}

sealed trait NominalAttr extends Attribute {
  def levels: Seq[ValType]
  override type ValType <: NominalValue
  type This <: NominalAttr
  def addLevel(level: ValType): This = {
    if(!levels.contains(level)) {
      unsafeAddLevel(level)
    } else {
      this.asInstanceOf[This]
    }
  }
  protected def unsafeAddLevel(level: ValType): This
}

case class StringAttribute(override val name: Symbol,
  override val levels: Seq[StringValue] = Nil,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override type ValType = StringValue
  override lazy val toWekaAttribute = new WekaAttribute(name.name, levels.to[FastVector], metadata)
  override type This = StringAttribute
  override protected def unsafeAddLevel(level: StringValue): This = StringAttribute(name, levels:+level, metadata)
}

case class NominalAttribute(override val name: Symbol,
  override val levels: Seq[SymbolValue],
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends NominalAttr {
  override type ValType = SymbolValue
  override lazy val toWekaAttribute = new WekaAttribute(name.name, levels.map(_.value.name).to[FastVector], metadata)
  override type This = NominalAttribute
  override protected def unsafeAddLevel(level: SymbolValue): This = NominalAttribute(name, levels:+level, metadata)
}

/* TODO find a way to deal with date attributes
sealed trait DateAttr extends Attribute {
  def dateFormat: String
  override type ValType = NominalValue
}

case class DateAttribute(override val name: Symbol, override val dateFormat: String,
  metadata: ProtectedProperties = new ProtectedProperties(new Properties())) extends DateAttr {
  override lazy val toWekaAttribute = new WekaAttribute(name.name, dateFormat, metadata)
}
* 
*/

sealed trait AttributeValue {
  type T
  def value: T
}
sealed trait NominalValue extends AttributeValue
case class StringValue(override val value: String) extends NominalValue {
  override type T = String
}
case class SymbolValue(override val value: Symbol) extends NominalValue {
  override type T = Symbol
}
case class NumericValue(override val value: Double) extends AttributeValue {
  override type T = Double
}

trait ConformType[+AV <: AttributeValue, +AD <: Attribute]
object ConformType {
  def apply[AV <: AttributeValue, AD <: Attribute](a: AV, ad: AD): Boolean = {
    (a, ad) match {
      case (_: NumericValue, _: NumericAttr) => true
      case (_: NominalValue, _: NominalAttr) => true
      case _ => false
    }
  }
}