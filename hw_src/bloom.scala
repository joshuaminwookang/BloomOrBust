// RoCC Bloom filter accelerator
// (c) 2019 Josh Kang and Andrew Thai

// Current version hard-codes the BF, m = 20,000 and k = 5
// To-do: parameterize m and k; make accelerator more scalable

package bloom

import Chisel._

import freechips.rocketchip.config._
import freechips.rocketchip.subsystem._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.rocket._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.tile._
import freechips.rocketchip.util.InOrderArbiter


class BloomAccel(opcodes: OpcodeSet)
(implicit p: Parameters) extends LazyRoCC(
    opcodes) {
  override lazy val module = new BloomAccelImp(this)
}

class BloomAccelImp(outer: BloomAccel)(implicit p: Parameters) extends LazyRoCCModuleImp(outer) {
  // accelerator memory 
  val bloom_bit_array = RegInit(Vec(Seq.fill(20000)(0.U(1.W))))
  val miss_counter = RegInit(0.U(64.W))
  // val busy = RegInit(Bool(false))

  // decode RoCC custom function
  val cmd = Queue(io.cmd)
  val funct = cmd.bits.inst.funct
  val hashed_string = cmd.bits.rs1
  val doInit = funct === UInt(0)
  val doMap = funct === UInt(1)
  val doTest = funct === UInt(2)

  // Hash computation hardware units
  val x0  = Wire(UInt())
  val y0  = Wire(UInt())

  val x1  = Wire(UInt())
  val y1  = Wire(UInt())

  val x2  = Wire(UInt())
  val y2  = Wire(UInt())

  val x3  = Wire(UInt())
  val y3  = Wire(UInt())

  val x4  = Wire(UInt())
  val y4  = Wire(UInt())

  val x5  = Wire(UInt())
  val y5  = Wire(UInt())

  // wiring logic for hashing
  x0 := hashed_string
  y0 := hashed_string >> 4

  x1 := (x0 + y0) % 20000.U(64.W)
  y1 := (y0 + 0.U(64.W)) % 20000.U(64.W)

  x2 := (x1 + y1) % 20000.U(64.W)
  y2 := (y1 + 1.U(64.W)) % 20000.U(64.W)

  x3 := (x2 + y2) % 20000.U(64.W)
  y3 := (y2 + 2.U(64.W)) % 20000.U(64.W)

  x4 := (x3 + y3) % 20000.U(64.W)
  y4 := (y3 + 3.U(64.W)) % 20000.U(64.W)

  x5 := (x4 + y4) % 20000.U(64.W)
  y5 := (y4 + 4.U(64.W)) % 20000.U(64.W)

  // logic for looking up elements in the bit array

  val found1 = Wire(UInt())
  val found2 = Wire(UInt())
  val found3 = Wire(UInt())
  val found4 = Wire(UInt())
  val found5 = Wire(UInt())

  found1 := bloom_bit_array(x1)
  found2 := bloom_bit_array(x2)
  found3 := bloom_bit_array(x3)
  found4 := bloom_bit_array(x4)
  found5 := bloom_bit_array(x5)

  // Custom function behaviors
  when (cmd.fire()) {
    when (doInit) {
      // BF_INIT
      bloom_bit_array := Reg(init = Vec.fill(20000)(0.U(1.W)))
      miss_counter := RegInit(0.U(64.W))
    }
    when (doMap) {
      // BF_MAP : map hash indices to array
      bloom_bit_array(x1) := 1.U(1.W)
      bloom_bit_array(x2) := 1.U(1.W)
      bloom_bit_array(x3) := 1.U(1.W)
      bloom_bit_array(x4) := 1.U(1.W)
      bloom_bit_array(x5) := 1.U(1.W)
    } 
    when (doTest) {
      // BF__TEST : hash and check if element was mapped before
      miss_counter := miss_counter + ~(found1 & found2 & found3 & found4 & found5)
    } 
  } 



  // PROCESSOR RESPONSE INTERFACE
  // Control for accelerator to communciate response back to host processor
  val doResp = cmd.bits.inst.xd
  val stallResp = doResp && !io.resp.ready 

  cmd.ready := !stallResp 
    // Command resolved if no stalls AND not issuing a load that will need a request
  io.resp.valid := cmd.valid && doResp 
    // Valid response if valid command, need a response, and no stalls
  io.resp.bits.rd := cmd.bits.inst.rd
    // Write to specified destination register address
  io.resp.bits.data := miss_counter
    // Send out miss counter data
  io.busy := cmd.valid 
    // Be busy when have pending memory requests or committed possibility of pending requests
  io.interrupt := Bool(false)
    // Set this true to trigger an interrupt on the processor (not the case for our current simplified implementation)
} 

// With this, the Rocket core will be compiled with BF accelerator as RoCC coprocessor
class WithBloomAccel extends Config ((site, here, up) => {

  case BuildRoCC => Seq(
    (p: Parameters) => {
      val bloom = LazyModule.apply(new BloomAccel(OpcodeSet.custom2)(p))
      bloom
    }
  )
})
