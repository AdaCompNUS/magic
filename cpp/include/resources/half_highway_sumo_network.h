#pragma once

#include <string>

namespace resources {

static const std::string HALF_HIGHWAY_SUMO_NETWORK =

R"(<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on Wed 23 Dec 2020 05:24:25 PM +08 by Eclipse SUMO netedit Version 1.7.0
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/netconvertConfiguration.xsd">

    <input>
        <sumo-net-file value="/home/leeyiyuan/summit/Data/highway.net.xml"/>
    </input>

    <output>
        <output-file value="/home/leeyiyuan/summit/Data/half_highway.net.xml"/>
    </output>

    <processing>
        <geometry.min-radius.fix.railways value="false"/>
        <geometry.max-grade.fix value="false"/>
        <offset.disable-normalization value="true"/>
        <lefthand value="true"/>
    </processing>

    <junctions>
        <no-internal-links value="true"/>
        <no-turnarounds value="true"/>
        <junctions.corner-detail value="5"/>
        <junctions.limit-turn-speed value="5.5"/>
        <rectangular-lane-cut value="false"/>
    </junctions>

    <pedestrian>
        <walkingareas value="false"/>
    </pedestrian>

    <report>
        <aggregate-warnings value="5"/>
    </report>

</configuration>
-->

<net version="1.6" junctionCornerDetail="5" lefthand="true" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

    <location netOffset="-11552344.72,-144514.59" convBoundary="0.00,0.00,589.58,349.35" origBoundary="103.776478,1.297372,103.781852,1.301223" projParameter="+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs"/>

    <type id="highway.bridleway" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.bus_guideway" priority="1" numLanes="1" speed="27.78" allow="bus" oneway="1"/>
    <type id="highway.cycleway" priority="1" numLanes="1" speed="8.33" allow="bicycle" oneway="0" width="1.00"/>
    <type id="highway.footway" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.ford" priority="1" numLanes="1" speed="2.78" allow="army" oneway="0"/>
    <type id="highway.living_street" priority="2" numLanes="1" speed="2.78" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.motorway" priority="14" numLanes="2" speed="39.44" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" oneway="1"/>
    <type id="highway.motorway_link" priority="9" numLanes="1" speed="22.22" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" oneway="1"/>
    <type id="highway.path" priority="1" numLanes="1" speed="2.78" allow="pedestrian bicycle" oneway="1" width="2.00"/>
    <type id="highway.pedestrian" priority="1" numLanes="1" speed="2.78" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.primary" priority="12" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.primary_link" priority="7" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.raceway" priority="15" numLanes="2" speed="83.33" allow="vip" oneway="0"/>
    <type id="highway.residential" priority="3" numLanes="1" speed="13.89" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.secondary" priority="11" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.secondary_link" priority="6" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.service" priority="1" numLanes="1" speed="5.56" allow="pedestrian delivery bicycle" oneway="0"/>
    <type id="highway.stairs" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.step" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.steps" priority="1" numLanes="1" speed="1.39" allow="pedestrian" oneway="1" width="2.00"/>
    <type id="highway.tertiary" priority="10" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.tertiary_link" priority="5" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.track" priority="1" numLanes="1" speed="5.56" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.trunk" priority="13" numLanes="2" speed="27.78" disallow="pedestrian bicycle tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.trunk_link" priority="8" numLanes="1" speed="22.22" disallow="pedestrian bicycle tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.unclassified" priority="4" numLanes="1" speed="13.89" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="highway.unsurfaced" priority="1" numLanes="1" speed="8.33" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
    <type id="railway.highspeed" priority="21" numLanes="1" speed="69.44" allow="rail_fast" oneway="1"/>
    <type id="railway.light_rail" priority="19" numLanes="1" speed="27.78" allow="rail_urban" oneway="1"/>
    <type id="railway.preserved" priority="16" numLanes="1" speed="27.78" allow="rail" oneway="1"/>
    <type id="railway.rail" priority="20" numLanes="1" speed="44.44" allow="rail" oneway="1"/>
    <type id="railway.subway" priority="18" numLanes="1" speed="27.78" allow="rail_urban" oneway="1"/>
    <type id="railway.tram" priority="17" numLanes="1" speed="13.89" allow="tram" oneway="1"/>

    <edge id="641237393" from="148118888" to="148118965" priority="14" type="highway.motorway" spreadType="center" shape="589.58,0.00 520.32,38.43 445.80,80.77 394.96,109.10 265.03,189.15 257.46,194.48 235.28,208.17 152.42,259.31 29.48,331.37 0.00,349.35">
        <lane id="641237393_0" index="0" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" speed="25.00" length="685.45" width="4.00" shape="586.67,-5.25 517.38,33.20 442.86,75.54 391.92,103.92 261.73,184.14 254.15,189.47 232.13,203.06 149.33,254.17 26.40,326.22 -3.12,344.23"/>
        <lane id="641237393_1" index="1" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" speed="25.00" length="685.45" width="4.00" shape="588.61,-1.75 519.34,36.69 444.82,79.03 393.95,107.37 263.93,187.48 256.36,192.81 234.23,206.47 151.39,257.60 28.45,329.65 -1.04,347.64"/>
        <lane id="641237393_2" index="2" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" speed="25.00" length="685.45" width="4.00" shape="590.55,1.75 521.30,40.17 446.78,82.51 395.97,110.83 266.13,190.82 258.56,196.15 236.33,209.87 153.45,261.02 30.51,333.09 1.04,351.06"/>
        <lane id="641237393_3" index="3" allow="private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2" speed="25.00" length="685.45" width="4.00" shape="592.49,5.25 523.26,43.66 448.74,86.00 398.00,114.28 268.33,194.16 260.77,199.49 238.43,213.28 155.51,264.45 32.56,336.52 3.12,354.47"/>
    </edge>

    <junction id="148118888" type="dead_end" x="589.58" y="0.00" incLanes="" intLanes="" shape="593.46,7.00 585.70,-7.00"/>
    <junction id="148118965" type="dead_end" x="0.00" y="349.35" incLanes="641237393_0 641237393_1 641237393_2 641237393_3" intLanes="" shape="-4.17,342.52 4.17,356.18"/>

</net>)";

}
