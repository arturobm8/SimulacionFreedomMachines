// ============================================================================
// Warehouse simulation compute shader
// One workgroup = one full simulation (3000 ticks)
// ============================================================================

// --- Constants --------------------------------------------------------------
const MAX_GRID: u32 = 90000u;    // 300*300
const MAX_ROBOTS: u32 = 70u;
const MAX_ORDERS: u32 = 600u;
const MAX_SHELVES: u32 = 50000u;
const MAX_STATIONS: u32 = 30u;
const MAX_ROUTE: u32 = 2048u;
const MAX_HEAP: u32 = 32768u;
const STRATEGY_WINDOW: u32 = 50u;
const GRID_WORDS: u32 = 22500u;  // MAX_GRID / 4
const WALK_WORDS: u32 = 2813u;   // ceil(MAX_GRID / 32)
const NO_VAL: u32 = 0xFFFFFFFFu;

// Robot states
const ST_IDLE: u32 = 0u;
const ST_TO_PICKUP: u32 = 1u;
const ST_TO_STATION: u32 = 2u;
const ST_RETURNING: u32 = 3u;

// --- Structs ----------------------------------------------------------------
struct SimParams {
    width: u32,
    height: u32,
    n_robots: u32,
    n_orders: u32,
    n_stations: u32,
    n_shelves: u32,
    ticks: u32,
    strategy_window: u32,
    stride_grid: u32,
    stride_shelves: u32,
    stride_heap: u32,
    _pad: u32,
}

struct Robot {
    pos_x: u32,
    pos_y: u32,
    state: u32,
    order_id: i32,
    shelf_home_x: i32,
    shelf_home_y: i32,
    station_dock_x: i32,
    station_dock_y: i32,
    route_len: u32,
    route_idx: u32,
    wait_ticks: u32,
    cells_moved: u32,
    busy_ticks: u32,
    order_index: i32,
    _pad0: u32,
    _pad1: u32,
}

struct Order {
    order_id: u32,
    shelf_id: u32,
    station_id: u32,
    tick_created: u32,
    tick_assigned: i32,
    tick_completed: i32,
}

struct Metrics {
    deadlocks: u32,
    completed_orders: u32,
    throughput_x1000: u32,
    avg_order_time_x100: u32,
    high_contention: u32,
    total_distance: u32,
    _pad0: u32,
    _pad1: u32,
}

// --- Bind group 0: simulation data ------------------------------------------
@group(0) @binding(0) var<storage, read>       params:        array<SimParams>;
@group(0) @binding(1) var<storage, read>       grids:         array<u32>;
@group(0) @binding(2) var<storage, read>       walkable:      array<u32>;
@group(0) @binding(3) var<storage, read_write> robots:        array<Robot>;
@group(0) @binding(4) var<storage, read_write> orders:        array<Order>;
@group(0) @binding(5) var<storage, read>       shelves_xy:    array<u32>;
@group(0) @binding(6) var<storage, read>       pickup_xy:     array<u32>;
@group(0) @binding(7) var<storage, read>       stations_xy:   array<u32>;

// --- Bind group 1: scratch + output -----------------------------------------
@group(1) @binding(0) var<storage, read_write> routes:        array<u32>;
@group(1) @binding(1) var<storage, read_write> cell_rsv:      array<u32>;
@group(1) @binding(2) var<storage, read_write> astar_g:       array<u32>;
@group(1) @binding(3) var<storage, read_write> astar_from:    array<u32>;
@group(1) @binding(4) var<storage, read_write> astar_heap:    array<vec4<u32>>;
@group(1) @binding(5) var<storage, read_write> pending:       array<u32>;
@group(1) @binding(6) var<storage, read_write> out_metrics:   array<Metrics>;

// --- Private globals (per-invocation) ---------------------------------------
var<private> g_sim: u32;
var<private> g_W: u32;
var<private> g_H: u32;
var<private> g_n_robots: u32;
var<private> g_n_orders: u32;
var<private> g_n_shelves: u32;
var<private> g_n_stations: u32;
var<private> g_ticks: u32;
var<private> g_strat_window: u32;

var<private> g_stride_grid: u32;
var<private> g_stride_shelves: u32;
var<private> g_stride_heap: u32;
var<private> g_grid_words: u32;
var<private> g_walk_words: u32;

var<private> g_cur_tick: u32;
var<private> g_pending_count: u32;
var<private> g_unreleased_ptr: u32;
var<private> g_astar_ver: u32;
var<private> g_heap_size: u32;

// Metrics accumulators
var<private> g_deadlocks: u32;
var<private> g_high_contention: u32;
var<private> g_completed: u32;
var<private> g_total_order_time: u32;

// --- Helpers ----------------------------------------------------------------
fn manhattan(ax: u32, ay: u32, bx: u32, by: u32) -> u32 {
    let dx = select(bx - ax, ax - bx, ax > bx);
    let dy = select(by - ay, ay - by, ay > by);
    return dx + dy;
}

fn grid_idx(x: u32, y: u32) -> u32 {
    return y * g_W + x;
}

fn get_cell(x: u32, y: u32) -> u32 {
    let idx = grid_idx(x, y);
    let base = g_sim * g_grid_words;
    let word = grids[base + idx / 4u];
    return (word >> ((idx % 4u) * 8u)) & 0xFFu;
}

fn is_walk(x: u32, y: u32) -> bool {
    let idx = grid_idx(x, y);
    let base = g_sim * g_walk_words;
    let word = walkable[base + idx / 32u];
    return ((word >> (idx % 32u)) & 1u) == 1u;
}

fn robot_base() -> u32 { return g_sim * MAX_ROBOTS; }
fn order_base() -> u32 { return g_sim * MAX_ORDERS; }
fn shelf_base() -> u32 { return g_sim * g_stride_shelves * 2u; }
fn pickup_base() -> u32 { return g_sim * g_stride_shelves * 2u; }
fn station_base() -> u32 { return g_sim * MAX_STATIONS * 2u; }
fn route_base(ri: u32) -> u32 { return (g_sim * MAX_ROBOTS + ri) * MAX_ROUTE; }
fn rsv_base() -> u32 { return g_sim * g_stride_grid; }
fn ag_base() -> u32 { return g_sim * g_stride_grid; }
fn af_base() -> u32 { return g_sim * g_stride_grid; }
fn ah_base() -> u32 { return g_sim * g_stride_heap; }
fn pend_base() -> u32 { return g_sim * MAX_ORDERS; }

fn get_shelf(shelf_id: u32) -> vec2<u32> {
    let b = shelf_base() + shelf_id * 2u;
    return vec2<u32>(shelves_xy[b], shelves_xy[b + 1u]);
}

fn get_pickup(shelf_id: u32) -> vec2<u32> {
    let b = pickup_base() + shelf_id * 2u;
    return vec2<u32>(pickup_xy[b], pickup_xy[b + 1u]);
}

fn get_station_dock(station_id: u32) -> vec2<u32> {
    let b = station_base() + station_id * 2u;
    return vec2<u32>(stations_xy[b], stations_xy[b + 1u]);
}

fn route_step(ri: u32, step: u32) -> vec2<u32> {
    let packed = routes[route_base(ri) + step];
    return vec2<u32>(packed >> 16u, packed & 0xFFFFu);
}

fn set_route_step(ri: u32, step: u32, x: u32, y: u32) {
    routes[route_base(ri) + step] = (x << 16u) | y;
}

// --- Cell reservation (tick-versioned, no clearing needed) ------------------
// Encoding: (tick << 8) | (robot_id + 1).  0 or wrong tick = free.
fn rsv_mark(x: u32, y: u32, tick: u32, rid: u32) {
    cell_rsv[rsv_base() + grid_idx(x, y)] = (tick << 8u) | (rid + 1u);
}

fn rsv_free(x: u32, y: u32, tick: u32) -> bool {
    let v = cell_rsv[rsv_base() + grid_idx(x, y)];
    return (v >> 8u) != tick;
}

// --- Min-heap for A* --------------------------------------------------------
fn heap_push(f: u32, g: u32, x: u32, y: u32) {
    let base = ah_base();
    if (g_heap_size >= g_stride_heap) { return; }
    var i = g_heap_size;
    g_heap_size = i + 1u;
    astar_heap[base + i] = vec4<u32>(f, g, x, y);
    // Bubble up
    while (i > 0u) {
        let pi = (i - 1u) / 2u;
        if (astar_heap[base + pi].x <= astar_heap[base + i].x) { break; }
        let tmp = astar_heap[base + pi];
        astar_heap[base + pi] = astar_heap[base + i];
        astar_heap[base + i] = tmp;
        i = pi;
    }
}

fn heap_pop() -> vec4<u32> {
    let base = ah_base();
    let top = astar_heap[base];
    g_heap_size = g_heap_size - 1u;
    if (g_heap_size == 0u) { return top; }
    astar_heap[base] = astar_heap[base + g_heap_size];
    var i: u32 = 0u;
    loop {
        let l = 2u * i + 1u;
        let r = 2u * i + 2u;
        var sm = i;
        if (l < g_heap_size && astar_heap[base + l].x < astar_heap[base + sm].x) { sm = l; }
        if (r < g_heap_size && astar_heap[base + r].x < astar_heap[base + sm].x) { sm = r; }
        if (sm == i) { break; }
        let tmp = astar_heap[base + sm];
        astar_heap[base + sm] = astar_heap[base + i];
        astar_heap[base + i] = tmp;
        i = sm;
    }
    return top;
}

// --- A* pathfinding ---------------------------------------------------------
// Returns route length written to routes[ri], or 0 on failure.
fn astar_run(ri: u32, sx: u32, sy: u32, gx: u32, gy: u32) -> u32 {
    if (sx >= g_W || sy >= g_H || gx >= g_W || gy >= g_H) { return 0u; }
    if (!is_walk(sx, sy) || !is_walk(gx, gy)) { return 0u; }
    if (sx == gx && sy == gy) {
        set_route_step(ri, 0u, sx, sy);
        return 1u;
    }

    g_astar_ver = g_astar_ver + 1u;
    let ver = g_astar_ver;
    let gb = ag_base();
    let fb = af_base();

    // Init start
    astar_g[gb + grid_idx(sx, sy)] = (ver << 16u);
    g_heap_size = 0u;
    heap_push(manhattan(sx, sy, gx, gy), 0u, sx, sy);

    while (g_heap_size > 0u) {
        let top = heap_pop();
        let g_cur = top.y;
        let cx = top.z;
        let cy = top.w;

        if (cx == gx && cy == gy) {
            // Reconstruct: count length
            var cnt: u32 = 1u;
            var tx = gx; var ty = gy;
            while (tx != sx || ty != sy) {
                let p = astar_from[fb + grid_idx(tx, ty)];
                tx = p >> 16u;
                ty = p & 0xFFFFu;
                cnt = cnt + 1u;
                if (cnt > MAX_ROUTE) { return 0u; }
            }
            // Write in reverse order
            tx = gx; ty = gy;
            var wi = cnt - 1u;
            set_route_step(ri, wi, tx, ty);
            while (tx != sx || ty != sy) {
                let p = astar_from[fb + grid_idx(tx, ty)];
                tx = p >> 16u;
                ty = p & 0xFFFFu;
                wi = wi - 1u;
                set_route_step(ri, wi, tx, ty);
            }
            return cnt;
        }

        // Stale check
        let cidx = grid_idx(cx, cy);
        let stored = astar_g[gb + cidx];
        if ((stored >> 16u) != ver || (stored & 0xFFFFu) < g_cur) { continue; }

        // Expand 4 neighbours
        for (var d: u32 = 0u; d < 4u; d = d + 1u) {
            var nx_i: i32; var ny_i: i32;
            switch d {
                case 0u: { nx_i = i32(cx) + 1; ny_i = i32(cy); }
                case 1u: { nx_i = i32(cx) - 1; ny_i = i32(cy); }
                case 2u: { nx_i = i32(cx); ny_i = i32(cy) + 1; }
                default:  { nx_i = i32(cx); ny_i = i32(cy) - 1; }
            }
            if (nx_i < 0 || ny_i < 0 || u32(nx_i) >= g_W || u32(ny_i) >= g_H) { continue; }
            let nx = u32(nx_i);
            let ny = u32(ny_i);
            if (!is_walk(nx, ny)) { continue; }

            let new_g = g_cur + 1u;
            let nidx = grid_idx(nx, ny);
            let ns = astar_g[gb + nidx];
            if ((ns >> 16u) == ver && (ns & 0xFFFFu) <= new_g) { continue; }

            astar_g[gb + nidx] = (ver << 16u) | new_g;
            astar_from[fb + nidx] = (cx << 16u) | cy;
            heap_push(new_g + manhattan(nx, ny, gx, gy), new_g, nx, ny);
        }
    }
    return 0u;
}

// --- Order release ----------------------------------------------------------
fn release_orders(tick: u32) {
    let ob = order_base();
    let pb = pend_base();
    while (g_unreleased_ptr < g_n_orders) {
        let oi = g_unreleased_ptr;
        if (orders[ob + oi].tick_created > tick) { break; }
        // Add to pending
        pending[pb + g_pending_count] = oi;
        g_pending_count = g_pending_count + 1u;
        g_unreleased_ptr = g_unreleased_ptr + 1u;
    }
}

// --- Greedy assignment ------------------------------------------------------
fn assign_orders(tick: u32) {
    let rb = robot_base();
    let ob = order_base();
    let pb = pend_base();
    let window = min(g_strat_window, g_pending_count);
    if (window == 0u) { return; }

    for (var ri: u32 = 0u; ri < g_n_robots; ri = ri + 1u) {
        if (robots[rb + ri].state != ST_IDLE) { continue; }
        if (g_pending_count == 0u) { break; }

        let rx = robots[rb + ri].pos_x;
        let ry = robots[rb + ri].pos_y;

        var best_pi: i32 = -1;
        var best_dist: u32 = 0xFFFFFFFFu;
        var best_px: u32 = 0u;
        var best_py: u32 = 0u;

        let w = min(g_strat_window, g_pending_count);
        for (var j: u32 = 0u; j < w; j = j + 1u) {
            let oi = pending[pb + j];
            let o = orders[ob + oi];
            let pu = get_pickup(o.shelf_id);
            if (pu.x == NO_VAL) { continue; }

            let dist = manhattan(rx, ry, pu.x, pu.y);
            if (dist < best_dist) {
                best_dist = dist;
                best_pi = i32(j);
                best_px = pu.x;
                best_py = pu.y;
            }
        }

        if (best_pi < 0) { continue; }

        // Run A* to pickup
        let rlen = astar_run(ri, rx, ry, best_px, best_py);
        if (rlen == 0u) { continue; }

        let oi = pending[pb + u32(best_pi)];
        let o = orders[ob + oi];

        // Update robot
        let sh = get_shelf(o.shelf_id);
        let dk = get_station_dock(o.station_id);
        robots[rb + ri].state = ST_TO_PICKUP;
        robots[rb + ri].order_id = i32(o.order_id);
        robots[rb + ri].order_index = i32(oi);
        robots[rb + ri].shelf_home_x = i32(sh.x);
        robots[rb + ri].shelf_home_y = i32(sh.y);
        robots[rb + ri].station_dock_x = i32(dk.x);
        robots[rb + ri].station_dock_y = i32(dk.y);
        robots[rb + ri].route_len = rlen;
        robots[rb + ri].route_idx = 0u;

        // Mark order assigned
        orders[ob + oi].tick_assigned = i32(tick);

        // Remove from pending (swap with last)
        g_pending_count = g_pending_count - 1u;
        if (u32(best_pi) < g_pending_count) {
            pending[pb + u32(best_pi)] = pending[pb + g_pending_count];
        }
    }
}

// --- State transitions on route completion ----------------------------------
fn plan_next_leg(ri: u32) {
    let rb = robot_base();
    let r = robots[rb + ri];
    if (r.route_len == 0u || r.route_idx != r.route_len - 1u) { return; }

    if (r.state == ST_TO_PICKUP) {
        // Transition to station
        let rlen = astar_run(ri, r.pos_x, r.pos_y, u32(r.station_dock_x), u32(r.station_dock_y));
        if (rlen > 0u) {
            robots[rb + ri].state = ST_TO_STATION;
            robots[rb + ri].route_len = rlen;
            robots[rb + ri].route_idx = 0u;
        }
    } else if (r.state == ST_TO_STATION) {
        // Transition to returning (go to shelf pickup cell)
        if (r.shelf_home_x < 0) { return; }
        let shelf_id = orders[order_base() + u32(r.order_index)].shelf_id;
        let pu = get_pickup(shelf_id);
        if (pu.x == NO_VAL) { return; }
        let rlen = astar_run(ri, r.pos_x, r.pos_y, pu.x, pu.y);
        if (rlen > 0u) {
            robots[rb + ri].state = ST_RETURNING;
            robots[rb + ri].route_len = rlen;
            robots[rb + ri].route_idx = 0u;
        }
    } else if (r.state == ST_RETURNING) {
        // Complete order and go idle
        if (r.order_index >= 0) {
            let ob = order_base();
            let oi = u32(r.order_index);
            orders[ob + oi].tick_completed = i32(g_cur_tick);
            g_completed = g_completed + 1u;
        }
        robots[rb + ri].state = ST_IDLE;
        robots[rb + ri].order_id = -1;
        robots[rb + ri].order_index = -1;
        robots[rb + ri].shelf_home_x = -1;
        robots[rb + ri].shelf_home_y = -1;
        robots[rb + ri].station_dock_x = -1;
        robots[rb + ri].station_dock_y = -1;
        robots[rb + ri].route_len = 0u;
        robots[rb + ri].route_idx = 0u;
    }
}

// --- Main entry point -------------------------------------------------------
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    g_sim = wg.x;
    // Keep all bindings alive (prevent compiler from optimizing them out)
    _ = grids[0];
    let p = params[g_sim];
    g_W = p.width;
    g_H = p.height;
    g_n_robots = p.n_robots;
    g_n_orders = p.n_orders;
    g_n_shelves = p.n_shelves;
    g_n_stations = p.n_stations;
    g_ticks = p.ticks;
    g_strat_window = p.strategy_window;
    g_stride_grid = p.stride_grid;
    g_stride_shelves = p.stride_shelves;
    g_stride_heap = p.stride_heap;
    g_grid_words = (g_stride_grid + 3u) / 4u;
    g_walk_words = (g_stride_grid + 31u) / 32u;

    g_pending_count = 0u;
    g_unreleased_ptr = 0u;
    g_astar_ver = 0u;
    g_deadlocks = 0u;
    g_high_contention = 0u;
    g_completed = 0u;
    g_total_order_time = 0u;

    let rb = robot_base();

    // Main tick loop
    for (var tick: u32 = 0u; tick < g_ticks; tick = tick + 1u) {
        g_cur_tick = tick;

        // 1. Release orders
        release_orders(tick);

        // 2. Assign orders (greedy)
        assign_orders(tick);

        let next_tick = tick + 1u;

        // 3. Propose movements
        // Store proposals in route_step or local. Use cell_rsv for resolution.
        // Proposals: for each robot, compute where it wants to go.

        // Phase A: plan next leg + compute proposals
        // We store proposals as packed (x << 16 | y) in a compact way.
        // Since we process robots sequentially, we can compute proposal inline.

        var anyone_moved = false;

        for (var ri: u32 = 0u; ri < g_n_robots; ri = ri + 1u) {
            let r = robots[rb + ri];

            if (r.state != ST_IDLE) {
                robots[rb + ri].busy_ticks = r.busy_ticks + 1u;
            }

            plan_next_leg(ri);

            // Re-read after plan_next_leg may have changed state
            let r2 = robots[rb + ri];

            var prop_x = r2.pos_x;
            var prop_y = r2.pos_y;

            if (r2.state != ST_IDLE && r2.route_len > 0u && r2.route_idx < r2.route_len - 1u) {
                let ns = route_step(ri, r2.route_idx + 1u);
                prop_x = ns.x;
                prop_y = ns.y;
            }

            // Resolve collision
            if (prop_x == r2.pos_x && prop_y == r2.pos_y) {
                // Staying — reserve cell
                rsv_mark(r2.pos_x, r2.pos_y, next_tick, ri);
            } else if (rsv_free(prop_x, prop_y, next_tick)) {
                // Move succeeds
                rsv_mark(prop_x, prop_y, next_tick, ri);
                robots[rb + ri].pos_x = prop_x;
                robots[rb + ri].pos_y = prop_y;
                robots[rb + ri].route_idx = r2.route_idx + 1u;
                robots[rb + ri].cells_moved = r2.cells_moved + 1u;
                anyone_moved = true;
            } else {
                // Blocked
                robots[rb + ri].wait_ticks = r2.wait_ticks + 1u;
                g_high_contention = g_high_contention + 1u;
                rsv_mark(r2.pos_x, r2.pos_y, next_tick, ri);
            }
        }

        // 4. Deadlock detection
        if (!anyone_moved) {
            var has_busy = false;
            for (var ri: u32 = 0u; ri < g_n_robots; ri = ri + 1u) {
                if (robots[rb + ri].state != ST_IDLE) { has_busy = true; break; }
            }
            if (has_busy) {
                g_deadlocks = g_deadlocks + 1u;
            }
        }
    }

    // Compute final metrics from order completion data
    let ob = order_base();
    var completed_count: u32 = 0u;
    var total_time: u32 = 0u;
    for (var i: u32 = 0u; i < g_n_orders; i = i + 1u) {
        if (orders[ob + i].tick_completed >= 0) {
            completed_count = completed_count + 1u;
            total_time = total_time + u32(orders[ob + i].tick_completed) - orders[ob + i].tick_created;
        }
    }

    // Total distance
    var total_dist: u32 = 0u;
    for (var ri: u32 = 0u; ri < g_n_robots; ri = ri + 1u) {
        total_dist = total_dist + robots[rb + ri].cells_moved;
    }

    // Write metrics
    var m: Metrics;
    m.deadlocks = g_deadlocks;
    m.completed_orders = completed_count;
    m.high_contention = g_high_contention;
    m.total_distance = total_dist;

    // throughput = completed / (ticks / 1000)  → completed * 1000 / ticks
    if (g_ticks > 0u) {
        m.throughput_x1000 = completed_count * 1000000u / g_ticks;  // ×1000 of (completed / (ticks/1000))
    } else {
        m.throughput_x1000 = 0u;
    }

    if (completed_count > 0u) {
        m.avg_order_time_x100 = total_time * 100u / completed_count;
    } else {
        m.avg_order_time_x100 = 9999900u;
    }

    m._pad0 = 0u;
    m._pad1 = 0u;

    out_metrics[g_sim] = m;
}
