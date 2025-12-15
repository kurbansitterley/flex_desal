from pyomo.environ import (
    ConcreteModel,
    Expression,
    value,
    assert_optimal_termination,
    units as pyunits,
    value,
    Block,
    Set,
    TransformationFactory,
)
from pyomo.network import Arc

from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import (
    MixingType,
    MomentumMixingType,
    Mixer,
    Separator,
    StateJunction,
)
import idaes.core.util.scaling as iscale
from idaes.core.util.scaling import (
    constraint_scaling_transform,
    calculate_scaling_factors,
    set_scaling_factor,
)

from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    PressureChangeType,
    MassTransferCoefficient,
    ConcentrationPolarizationType,
)
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock
from watertap.unit_models.pressure_changer import Pump
from watertap.core.solvers import get_solver


from wrd.components.pump import *
from wrd.utilities import load_config, get_config_value, get_config_file
from models.head_loss import HeadLoss
from wrd.components.ro import *
from wrd.components.ro_skid import *
from srp.utils import touch_flow_and_conc

__all__ = [
    "build_ro_train",
    "set_ro_train_op_conditions",
    "set_ro_train_scaling",
    "initialize_ro_train",
    "report_ro_train",
]


def build_system():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = NaClParameterBlock()

    m.fs.feed = Feed(property_package=m.fs.properties)

    m.fs.ro_train = FlowsheetBlock(dynamic=False)
    build_ro_train(m.fs.ro_train, prop_package=m.fs.properties)

    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.brine = Product(property_package=m.fs.properties)

    # Arcs to connect the unit models
    m.fs.feed_to_train = Arc(
        source=m.fs.feed.outlet,
        destination=m.fs.ro_train.feed.inlet,
    )
    m.fs.train_to_product = Arc(
        source=m.fs.ro_train.product.outlet,
        destination=m.fs.product.inlet,
    )
    m.fs.train_to_brine = Arc(
        source=m.fs.ro_train.disposal.outlet,
        destination=m.fs.brine.inlet,
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-1, index=("Liq", "H2O")  # changed from 1
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
    )

    return m


def set_inlet_conditions(m, Qin=2637, Cin=0.5, file="wrd_ro_inputs_8_19_21.yaml"):

    config_data = load_config(get_config_file(file))

    Pout = get_config_value(config_data, "pump_outlet_pressure", "pumps", f"pump_1")

    m.fs.feed.properties.calculate_state(
        var_args={
            ("flow_vol_phase", ("Liq")): Qin * pyunits.gallons / pyunits.minute,
            ("conc_mass_phase_comp", ("Liq", "NaCl")): Cin * pyunits.g / pyunits.L,
            ("pressure", None): Pout,
            ("temperature", None): 273.15 + 27,
        },
        hold_state=True,
    )


def build_ro_train(
    blk, num_stages=3, file="wrd_ro_inputs_8_19_21.yaml", prop_package=None
):
    m = blk.model()
    if prop_package is None:
        prop_package = m.fs.properties

    blk.config_data = load_config(get_config_file(file))

    blk.stages = Set(initialize=range(1, num_stages + 1))
    blk.skids = FlowsheetBlock(blk.stages, dynamic=False)

    blk.feed = StateJunction(property_package=prop_package)
    touch_flow_and_conc(blk.feed)

    blk.mixer = Mixer(
        property_package=prop_package,
        inlet_list=[f"skid_{i}_to_product" for i in blk.stages],
        momentum_mixing_type=MomentumMixingType.none,
    )
    touch_flow_and_conc(blk.mixer)

    blk.product = StateJunction(property_package=prop_package)
    blk.disposal = StateJunction(property_package=prop_package)
    touch_flow_and_conc(blk.product)
    touch_flow_and_conc(blk.disposal)
    blk.recovery_vol = Expression(
        expr=blk.product.properties[0].flow_vol_phase["Liq"]
        / blk.feed.properties[0].flow_vol_phase["Liq"]
    )

    for i in blk.stages:
        # print(f"Building skid {i}\n\n\n")
        build_ro_skid(blk.skids[i], stage_num=i, file=file, prop_package=prop_package)
    for i in blk.stages:
        # print(i)
        # print()
        if i == blk.stages.first():
            ain = Arc(source=blk.feed.outlet, destination=blk.skids[i].feed.inlet)
            blk.add_component(f"feed_to_skid_{i}", ain)
            # print(f"feed_to_skid_{i}")
            aout = Arc(
                source=blk.skids[i].disposal.outlet,
                destination=blk.skids[i + 1].feed.inlet,
            )
            blk.add_component(f"skid_{i}_to_skid_{i+1}", aout)
            # print(f"skid_{i}_to_skid_{i+1}")
        elif i == blk.stages.last():
            # aout_prod = Arc(
            #     source=blk.skids[i].product.outlet, destination=blk.product.inlet
            # )
            # blk.add_component(f"skid_{i}_to_product", aout_prod)
            # print(f"skid_{i}_to_product")
            aout_brine = Arc(
                source=blk.skids[i].disposal.outlet, destination=blk.disposal.inlet
            )
            blk.add_component(f"skid_{i}_to_brine", aout_brine)
            # print(f"skid_{i}_to_brine")
        else:
            aout = Arc(
                source=blk.skids[i].disposal.outlet,
                destination=blk.skids[i + 1].feed.inlet,
            )
            blk.add_component(f"skid_{i}_to_skid_{i+1}", aout)
            # print(f"skid_{i}_to_skid_{i+1}")
        mix_in = blk.mixer.find_component(f"skid_{i}_to_product")
        mix_arc = Arc(source=blk.skids[i].product.outlet, destination=mix_in)
        # ap = Arc(
        #     source=blk.skids[i].product.outlet,
        blk.add_component(f"skid_{i}_to_product", mix_arc)
        # print(f"skid_{i}_to_mixer")

    blk.mixer_to_product = Arc(source=blk.mixer.outlet, destination=blk.product.inlet)

    TransformationFactory("network.expand_arcs").apply_to(blk)


def set_ro_train_scaling(blk):

    for i in blk.stages:
        set_ro_skid_scaling(blk.skids[i])


def set_ro_train_op_conditions(blk):

    for i in blk.stages:
        set_ro_skid_op_conditions(blk.skids[i])
        # print(f"dof skid {i}: {degrees_of_freedom(blk.skids[i])}")

    blk.mixer.outlet.pressure[0].fix(101325)


def initialize_ro_train(blk):
    for i in blk.stages:
        if i == blk.stages.first():
            a = blk.find_component(f"feed_to_skid_{i}")
            propagate_state(a)
            initialize_ro_skid(blk.skids[i])
            a = blk.find_component(f"skid_{i}_to_skid_{i+1}")
            propagate_state(a)
        elif i == blk.stages.last():
            initialize_ro_skid(blk.skids[i])
            a = blk.find_component(f"skid_{i}_to_product")
            propagate_state(a)
            a = blk.find_component(f"skid_{i}_to_brine")
        else:
            initialize_ro_skid(blk.skids[i])
            a = blk.find_component(f"skid_{i}_to_skid_{i+1}")
            propagate_state(a)
        a = blk.find_component(f"skid_{i}_to_product")
        propagate_state(a)

    blk.mixer.initialize()
    propagate_state(blk.mixer_to_product)
    blk.product.initialize()


def initialize_system(m):
    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_train)

    initialize_ro_train(m.fs.ro_train)

    propagate_state(m.fs.train_to_product)
    m.fs.product.initialize()
    propagate_state(m.fs.train_to_brine)
    m.fs.brine.initialize()


def report_ro_train(blk, train_num=None, w=30):
    if train_num is None:
        title = "RO Train Report"
    else:
        title = f"RO Train {train_num} Report"
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "=" * side + f" {title} " + "=" * side
    print(f"\n{header}\n")
    for i in blk.stages:
        title = f"Stage {i}"
        side = int(((3 * w) - len(title)) / 2) - 1
        header = "_" * side + f" {title} " + "_" * side
        print(f"\n\n{header}\n")
        report_ro_skid(blk.skids[i], w=w)

    title = f"Overall Train Performance"
    print(f'{"Parameter":<{w}s}{"Value":<{w}s}{"Units":<{w}s}')
    print(f"{'-' * (3 * w)}")
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "_" * side + f" {title} " + "_" * side
    print(f"\n\n{header}\n")
    for i, inlet in enumerate(blk.mixer.config.inlet_list, 1):
        sb = blk.mixer.find_component(f"{inlet}_state")
        print(
            f'{f"Skid {i} Perm Conc":<{w}s}{value(pyunits.convert(sb[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg / pyunits.L)):<{w}.3f}{"mg/L"}'
        )
        print(
            f'{f"Skid {i} Perm Flow":<{w}s}{value(pyunits.convert(sb[0].flow_vol_phase["Liq"], to_units=pyunits.gallons / pyunits.minute)):<{w}.3f}{"gpm"}'
        )
    print(
        f'{f"Total Perm Flow":<{w}s}{value(pyunits.convert(blk.product.properties[0].flow_vol_phase["Liq"], to_units=pyunits.gallons / pyunits.minute)):<{w}.3f}{"gpm"}'
    )
    print(
        f'{f"Final Perm Conc":<{w}s}{value(pyunits.convert(blk.product.properties[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg / pyunits.L)):<{w}.3f}{"mg/L"}'
    )
    print(
        f'{f"Final Brine Conc":<{w}s}{value(pyunits.convert(blk.disposal.properties[0].conc_mass_phase_comp["Liq", "NaCl"], to_units=pyunits.mg / pyunits.L)):<{w}.3f}{"mg/L"}'
    )
    print(f'{f"Overall Recovery":<{w}s}{value(blk.recovery_vol)*100:<{w}.3f}{"%"}')

    # tot_flow_in = sum(
    #     value(
    #         pyunits.convert(
    #             u.find_component(f"{x}_state")[0].flow_vol_phase["Liq"],
    #             to_units=pyunits.gallons / pyunits.minute,
    #         )
    #     )
    #     for x in u.config.inlet_list
    # )
    # print(
    #     f'{"TOTAL INLET FLOW":<{w}s}{f"{tot_flow_in:<{w},.1f}"}{"gpm":<{w}s}'
    # )
    # for x in u.config.inlet_list:
    #     sb = u.find_component(f"{x}_state")
    #     flow_in = value(
    #         pyunits.convert(
    #             sb[0].flow_vol_phase["Liq"],
    #             to_units=pyunits.gallons / pyunits.minute,
    #         )
    #     )
    #     conc_in = value(
    #         pyunits.convert(
    #             sb[0].conc_mass_phase_comp["Liq", "TDS"],
    #             to_units=pyunits.mg / pyunits.L,
    #         )
    #     )
    #     print(
    #         f'{"   Flow " + x.replace("_", " ").title():<{w}s}{f"{flow_in:<{w},.1f}"}{"gpm":<{w}s}'
    #     )
    #     print(
    #         f'{"   TDS " + x.replace("_", " ").title():<{w}s}{f"{conc_in:<{w},.1f}"}{"mg/L":<{w}s}'
    #     )
    # flow_out = value(
    #     pyunits.convert(
    #         ms[0].flow_vol_phase["Liq"],
    #         to_units=pyunits.gallons / pyunits.minute,
    #     )
    # )
    # conc_out = value(
    #     pyunits.convert(
    #         ms[0].conc_mass_phase_comp["Liq", "TDS"],
    #         to_units=pyunits.mg / pyunits.L,
    #     )
    # )
    # print(f'{"Outlet Flow":<{w}s}{f"{flow_out:<{w},.1f}"}{"gpm":<{w}s}')
    # print(f'{"Outlet TDS":<{w}s}{f"{conc_out:<{w},.1f}"}{"mg/L":<{w}s}')


if __name__ == "__main__":
    m = build_system()
    set_ro_train_scaling(m.fs.ro_train)
    m.fs.feed.properties[0].conc_mass_phase_comp
    m.fs.feed.properties[0].flow_vol_phase
    calculate_scaling_factors(m)
    set_inlet_conditions(m)
    set_ro_train_op_conditions(m.fs.ro_train)
    # initialize_ro_train(m.fs.ro_train)
    initialize_system(m)
    solver = get_solver()
    results = solver.solve(m, tee=True)
    # report_ro_skid(m.fs.ro_train.skids[1], w=30)
    report_ro_train(m.fs.ro_train, w=30)
    # d = m.fs.ro_train.skids[1].disposal.properties[0].define_state_vars()
    # import pprint
    # pprint.pprint(d)
    # for name, v in d.items():
    #     if v.is_indexed():
    #         for i, vv in v.items():
    #             print(f"{name} {i}: {value(vv)}")
    #     else:
    #         print(f"{name}: {value(v)}")
    print(f"Degrees of freedom: {degrees_of_freedom(m)}")

    # m.fs.ro_train.skids[1].feed.properties[0].display()
