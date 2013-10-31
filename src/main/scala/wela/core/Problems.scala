package wela.core


case class Problem[+L <: Attribute](name: String, label: L) {
  def withAttributes[A <: Attribute](attr: A*): ProblemWithAttributes[L, List[A]] = {
    new ProblemWithAttributes(name, attr.toList, label)
  }
}

trait AbstractProblemWithAttributes[+L <: Attribute, +AS <: List[Attribute]] {
  def name: String

  def attrs: AS

  def label: L

  val attrDefinitions = (label :: attrs).map(a => a.name -> a).toMap

  def withInstances(inst: Instance*): AbstractDataset[L, AS]

  def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedProblemWithAttributes[L, AS]

  def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedProblemWithAttributes[L, AS]

  def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedProblemWithAttributes[L, AS]
}

class ProblemWithAttributes[+L <: Attribute, +AS <: List[Attribute]] protected[core](override val name: String, override val attrs: AS, override val label: L) extends AbstractProblemWithAttributes[L, AS] {
  override def withInstances(inst: Instance*): Dataset[L, AS] = {
    new Dataset(this, inst)
  }

  override def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedProblemWithAttributes[L, AS] = {
    val mapper = new ProblemMapping(attr, newAttr, f)(comp)
    new MappedProblemWithAttributes[L,AS](name, attrs, label, List(mapper))
  }

  override def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedProblemWithAttributes[L, AS] = {
    if (!newLabel.isDefined) require(attr != label.name, "you can't replace the training label/class unless you specify a replacement")
    val mapper = new ProblemMultiMapper(attr, newAttributes, f, newLabel)
    new MappedProblemWithAttributes[L, AS](name, attrs, label, List(mapper))
  }

  def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedProblemWithAttributes[L, AS] = {
    val mapper = new ProblemDynMapper(attr, newAttrPrefix, f)
    new MappedProblemWithAttributes[L, AS](name, attrs, label, List(mapper))
  }
}

class MappedProblemWithAttributes[+L <: Attribute, +AS <: List[Attribute]] protected[core](override val name: String, override val attrs: AS, override val label: L, protected[core] val mappings: List[Mapper]) extends AbstractProblemWithAttributes[L, AS] {
  override def withInstances(inst: Instance*): MappedDataset[L, AS] = {
    new MappedDataset(this, inst)
  }

  override def withMapping[AD <: Attribute](attr: Symbol, newAttr: AD)(f: AttributeValue => newAttr.ValType)(implicit comp: Compatible[AD, newAttr.ValType]): MappedProblemWithAttributes[L, AS] = {
    val mapper = new ProblemMapping(attr, newAttr, f)(comp)
    new MappedProblemWithAttributes[L,AS](name,attrs,label, mapper :: mappings)
  }

  override def withFlatMapping[AD <: Attribute](attr: Symbol, newAttributes: Seq[AD], newLabel: Option[Symbol] = None)(f: AttributeValue => Seq[(Symbol, AttributeValue)]): MappedProblemWithAttributes[L, AS] = {
    if (!newLabel.isDefined) require(attr != label.name)
    val mapper = new ProblemMultiMapper(attr, newAttributes, f, newLabel)
    new MappedProblemWithAttributes[L,AS](name,attrs,label, mapper :: mappings)
  }

  override def explodeAttributes(attr: Symbol, newAttrPrefix: String)(f: AttributeValue => Seq[String]): MappedProblemWithAttributes[L, AS] = {
    val mapper = new ProblemDynMapper(attr, newAttrPrefix, f)
    new MappedProblemWithAttributes[L,AS](name,attrs,label, mapper :: mappings)
  }


}