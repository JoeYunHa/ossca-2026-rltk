bracket_terminal::add_wasm_support!();
use bracket_pathfinding::prelude::*;
use bracket_random::prelude::*;
use bracket_terminal::prelude::*;

// Map dimensions: leave rows 48-49 for the HUD
const MAP_W: i32 = 80;
const MAP_H: i32 = 48;
const TORCH_RADIUS: i32 = 8;
const MAX_LEVEL: u32 = 5;
/// Base monster move interval in milliseconds; decreases by this amount each level.
const MONSTER_BASE_MS: f32 = 500.0;
const MONSTER_SPEED_STEP_MS: f32 = 100.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Decompose a flat map index into (x, y).
fn xy(pos: usize) -> (i32, i32) {
    (pos as i32 % MAP_W, pos as i32 / MAP_W)
}

/// Pad `s` to exactly 80 characters for the HUD row.
/// Call at log-write time so render never allocates.
fn hud(s: &str) -> String {
    format!("{s:<80}")
}

/// Return a random passable neighbour of `pos`, or `pos` itself if surrounded.
fn random_step(pos: usize, map: &Map, rng: &mut RandomNumberGenerator) -> usize {
    let (x, y) = xy(pos);
    let mut candidates = [0usize; 4];
    let mut count = 0usize;
    for &(dx, dy) in &[(0i32, -1i32), (0, 1), (-1, 0), (1, 0)] {
        let nx = x + dx;
        let ny = y + dy;
        if (0..MAP_W).contains(&nx) && (0..MAP_H).contains(&ny) {
            let nidx = map.idx(nx, ny);
            if map.tiles[nidx] != TileType::Wall {
                candidates[count] = nidx;
                count += 1;
            }
        }
    }
    if count == 0 {
        pos
    } else {
        candidates[rng.range(0, count as i32) as usize]
    }
}

// ── Tile types ────────────────────────────────────────────────────────────────

#[derive(PartialEq, Copy, Clone)]
enum TileType {
    Wall,
    Floor,
    Exit,
}

// ── Map ───────────────────────────────────────────────────────────────────────

struct Map {
    tiles: Vec<TileType>,
    revealed: Vec<bool>,
    visible: Vec<bool>,
}

impl Map {
    fn new() -> Self {
        let size = (MAP_W * MAP_H) as usize;
        Self {
            tiles: vec![TileType::Wall; size],
            revealed: vec![false; size],
            visible: vec![false; size],
        }
    }

    fn idx(&self, x: i32, y: i32) -> usize {
        debug_assert!(
            (0..MAP_W).contains(&x) && (0..MAP_H).contains(&y),
            "idx({x},{y}) out of bounds"
        );
        (y * MAP_W + x) as usize
    }

    fn carve_room(&mut self, room: &Rect) {
        for y in (room.y1 + 1)..room.y2 {
            for x in (room.x1 + 1)..room.x2 {
                let i = self.idx(x, y);
                self.tiles[i] = TileType::Floor;
            }
        }
    }

    fn carve_h_tunnel(&mut self, x1: i32, x2: i32, y: i32) {
        for x in i32::min(x1, x2)..=i32::max(x1, x2) {
            let i = self.idx(x, y);
            self.tiles[i] = TileType::Floor;
        }
    }

    fn carve_v_tunnel(&mut self, y1: i32, y2: i32, x: i32) {
        for y in i32::min(y1, y2)..=i32::max(y1, y2) {
            let i = self.idx(x, y);
            self.tiles[i] = TileType::Floor;
        }
    }
}

impl Algorithm2D for Map {
    fn dimensions(&self) -> Point {
        Point::new(MAP_W, MAP_H)
    }
}

impl BaseMap for Map {
    fn is_opaque(&self, idx: usize) -> bool {
        self.tiles[idx] == TileType::Wall
    }

    fn get_available_exits(&self, idx: usize) -> SmallVec<[(usize, f32); 10]> {
        let mut exits = SmallVec::new();
        let (x, y) = xy(idx);
        let w = MAP_W as usize;

        if x > 0 && self.tiles[idx - 1] != TileType::Wall {
            exits.push((idx - 1, 1.0));
        }
        if x < MAP_W - 1 && self.tiles[idx + 1] != TileType::Wall {
            exits.push((idx + 1, 1.0));
        }
        if y > 0 && self.tiles[idx - w] != TileType::Wall {
            exits.push((idx - w, 1.0));
        }
        if y < MAP_H - 1 && self.tiles[idx + w] != TileType::Wall {
            exits.push((idx + w, 1.0));
        }
        exits
    }

    fn get_pathing_distance(&self, idx1: usize, idx2: usize) -> f32 {
        let (x1, y1) = xy(idx1);
        let (x2, y2) = xy(idx2);
        DistanceAlg::Pythagoras.distance2d(Point::new(x1, y1), Point::new(x2, y2))
    }
}

// ── Entities ──────────────────────────────────────────────────────────────────

struct Monster {
    pos: usize,
    /// True while the monster has direct line-of-sight to the player.
    alerted: bool,
    /// Last tile where the player was spotted; drives search behaviour after LoS breaks.
    last_known_pos: Option<usize>,
}

// ── Game state ────────────────────────────────────────────────────────────────

#[derive(PartialEq)]
enum RunState {
    Running,
    LevelClear,
    GameClear,
    Lost,
}

struct State {
    map: Map,
    player_pos: usize,
    monsters: Vec<Monster>,
    run_state: RunState,
    log: String,
    level: u32,
    /// Accumulated time since last monster move, in milliseconds.
    monster_timer: f32,
    rng: RandomNumberGenerator,
    /// Map index of the key; None once the player has picked it up.
    key_pos: Option<usize>,
    has_key: bool,
}

impl State {
    fn build(level: u32) -> Self {
        let mut map = Map::new();
        let mut rng = RandomNumberGenerator::new();
        let mut rooms: Vec<Rect> = Vec::new();

        // Generate rooms — attempt many placements to fill the map
        for _ in 0..30 {
            let w = rng.range(4, 11);
            let h = rng.range(4, 11);
            let x = rng.range(1, MAP_W - w - 1);
            let y = rng.range(1, MAP_H - h - 1);
            let room = Rect::with_size(x, y, w, h);

            if rooms.iter().all(|r| !r.intersect(&room)) {
                map.carve_room(&room);
                if let Some(prev) = rooms.last() {
                    let c1 = prev.center();
                    let c2 = room.center();
                    if rng.range(0, 2) == 0 {
                        map.carve_h_tunnel(c1.x, c2.x, c1.y);
                        map.carve_v_tunnel(c1.y, c2.y, c2.x);
                    } else {
                        map.carve_v_tunnel(c1.y, c2.y, c1.x);
                        map.carve_h_tunnel(c1.x, c2.x, c2.y);
                    }
                }
                rooms.push(room);
                // Ensure enough rooms for all monsters plus start and exit
                if rooms.len() >= (MAX_LEVEL as usize) + 2 {
                    break;
                }
            }
        }

        // Retry if the RNG produced too few non-overlapping rooms (rare)
        if rooms.len() < 2 {
            return Self::build(level);
        }

        // Player starts at center of first room
        let start = rooms[0].center();
        let player_pos = map.idx(start.x, start.y);

        // Exit at center of last room
        let exit_center = rooms.last().unwrap().center();
        let exit_idx = map.idx(exit_center.x, exit_center.y);
        map.tiles[exit_idx] = TileType::Exit;

        // Place the key in the middle room (between start and exit).
        // If there are only two rooms, the key goes in the exit room area (player still
        // has to reach it before stepping on ">").
        let key_room_idx = rooms.len() / 2;
        let key_center = rooms[key_room_idx].center();
        let key_idx = map.idx(key_center.x, key_center.y);
        // Avoid placing the key on the exit tile itself.
        let key_pos = if key_idx != exit_idx {
            Some(key_idx)
        } else {
            // Fallback: use the room before the exit room.
            let fallback = rooms[rooms.len().saturating_sub(2)].center();
            let fidx = map.idx(fallback.x, fallback.y);
            if fidx != exit_idx { Some(fidx) } else { None }
        };

        // Place exactly `level` monsters in the non-start rooms nearest the exit.
        // Monsters guard the path to the exit; earlier rooms remain safer.
        let monsters = rooms
            .iter()
            .skip(1) // never in start room
            .rev() // start from rooms closest to exit
            .filter_map(|room| {
                let c = room.center();
                let idx = map.idx(c.x, c.y);
                if idx != exit_idx {
                    Some(Monster {
                        pos: idx,
                        alerted: false,
                        last_known_pos: None,
                    })
                } else {
                    None
                }
            })
            .take(level as usize)
            .collect();

        let mut state = State {
            map,
            player_pos,
            monsters,
            run_state: RunState::Running,
            log: hud(&format!(
                "Level {}/{}  Pick up key (k), reach exit (>), avoid goblins (g). [Arrow/WASD]",
                level, MAX_LEVEL
            )),
            level,
            monster_timer: 0.0,
            rng,
            key_pos,
            has_key: false,
        };
        // Initialise FOV before the first tick so tiles are visible immediately
        state.update_fov();
        state
    }

    fn try_move_player(&mut self, dx: i32, dy: i32) {
        let current_pos = self.player_pos;
        let (cx, cy) = xy(self.player_pos);
        let x = cx + dx;
        let y = cy + dy;
        if !(0..MAP_W).contains(&x) || !(0..MAP_H).contains(&y) {
            return;
        }
        let new_pos = self.map.idx(x, y);
        if self.map.tiles[new_pos] == TileType::Wall {
            return;
        }
        self.player_pos = new_pos;

        // Pick up the key if the player steps on it.
        if self.key_pos == Some(new_pos) {
            self.has_key = true;
            self.key_pos = None;
            self.log = hud("You picked up the key! Now reach the exit (>).");
        }

        if self.map.tiles[new_pos] == TileType::Exit {
            if !self.has_key {
                self.log = hud("The exit is locked — find the key (k) first!");
                // Step back: don't allow entry without the key.
                self.player_pos = current_pos;
                return;
            }
            if self.level < MAX_LEVEL {
                self.run_state = RunState::LevelClear;
                self.log = hud(&format!("Level {} cleared!", self.level));
            } else {
                self.run_state = RunState::GameClear;
                self.log = hud("All levels cleared! GAME COMPLETE!");
            }
        }
    }

    fn update_fov(&mut self) {
        for v in self.map.visible.iter_mut() {
            *v = false;
        }
        let (px, py) = xy(self.player_pos);
        let fov = field_of_view_set(Point::new(px, py), TORCH_RADIUS, &self.map);
        for p in &fov {
            let idx = self.map.idx(p.x, p.y);
            self.map.visible[idx] = true;
            self.map.revealed[idx] = true;
        }
    }

    /// Move every monster according to a three-tier AI:
    ///
    /// 1. **Alert** — monster is inside the player's torch FOV (mutual LoS).
    ///    Chases via A* and remembers the player's position.
    /// 2. **Search** — LoS just broke; monster moves toward the last known position.
    ///    Forgets and switches to wander once it reaches that tile.
    /// 3. **Wander** — no memory of the player; moves to a random passable neighbour.
    fn move_monsters(&mut self) {
        let player_pos = self.player_pos;

        // Stack-allocated update buffer — no heap allocation (count ≤ MAX_LEVEL).
        let mut updates: [(usize, bool, Option<usize>); MAX_LEVEL as usize] =
            [(0, false, None); MAX_LEVEL as usize];
        let n = self.monsters.len();

        for (i, update) in updates.iter_mut().enumerate().take(n) {
            // Extract Copy fields so we don't hold a borrow on self.monsters
            // while we need &mut self.rng below.
            let m_pos = self.monsters[i].pos;
            let m_last_known = self.monsters[i].last_known_pos;

            // LoS: the player's visible[] array is symmetric — a tile lit by the
            // torch is mutually visible, so if the monster stands on a visible tile
            // it can see the player.
            let can_see_player = self.map.visible[m_pos];

            *update = if can_see_player {
                // Chase: A* toward player, update memory.
                let path = a_star_search(m_pos, player_pos, &self.map);
                let next = if path.success && path.steps.len() > 1 {
                    path.steps[1]
                } else {
                    m_pos
                };
                (next, true, Some(player_pos))
            } else if let Some(target) = m_last_known {
                if m_pos == target {
                    // Reached last known tile, player gone — forget and wander.
                    (random_step(m_pos, &self.map, &mut self.rng), false, None)
                } else {
                    // Still heading toward memory target.
                    let path = a_star_search(m_pos, target, &self.map);
                    let next = if path.success && path.steps.len() > 1 {
                        path.steps[1]
                    } else {
                        m_pos
                    };
                    (next, false, Some(target))
                }
            } else {
                // Wander randomly.
                (random_step(m_pos, &self.map, &mut self.rng), false, None)
            };
        }

        for (m, &(pos, alerted, last_known)) in self.monsters.iter_mut().zip(updates[..n].iter()) {
            m.pos = pos;
            m.alerted = alerted;
            m.last_known_pos = last_known;
        }

        if self.monsters.iter().any(|m| m.pos == player_pos) {
            self.run_state = RunState::Lost;
            self.log = hud("A goblin caught you! GAME OVER");
        }
    }

    fn render_map(&self, draw_batch: &mut DrawBatch) {
        let (px, py) = xy(self.player_pos);
        let player_pt = Point::new(px, py);

        for (idx, tile) in self.map.tiles.iter().enumerate() {
            if !self.map.revealed[idx] {
                continue;
            }
            let (x, y) = xy(idx);

            let (glyph, fg) = if self.map.visible[idx] {
                let dist = DistanceAlg::Pythagoras.distance2d(player_pt, Point::new(x, y));
                let t = (1.0_f32 - (dist / TORCH_RADIUS as f32) * 0.75).max(0.1);
                match tile {
                    TileType::Wall => ("#", RGB::from_f32(0.6 * t, 0.6 * t, 0.55 * t)),
                    TileType::Floor => (".", RGB::from_f32(0.45 * t, 0.38 * t, 0.28 * t)),
                    TileType::Exit => (">", RGB::from_f32(0.0, t, 0.5 * t)),
                }
            } else {
                match tile {
                    TileType::Wall => ("#", RGB::from_f32(0.18, 0.18, 0.18)),
                    TileType::Floor => (".", RGB::from_f32(0.15, 0.15, 0.15)),
                    TileType::Exit => (">", RGB::from_f32(0.0, 0.25, 0.12)),
                }
            };

            draw_batch.print_color(
                Point::new(x, y),
                glyph,
                ColorPair::new(fg, RGB::named(BLACK)),
            );
        }
    }

    fn render(&self, ctx: &mut BTerm) {
        let mut draw_batch = DrawBatch::new();
        draw_batch.cls();

        self.render_map(&mut draw_batch);

        // Render monsters:
        //   - Alerted (direct LoS to player): always visible, bright red
        //   - Non-alerted inside torch FOV: visible, dark red
        //   - Non-alerted outside torch FOV: hidden (wandering or searching)
        for m in &self.monsters {
            let in_fov = self.map.visible[m.pos];
            if m.alerted || in_fov {
                let color = if m.alerted {
                    RGB::named(RED)
                } else {
                    RGB::from_f32(0.55, 0.0, 0.0)
                };
                let (mx, my) = xy(m.pos);
                draw_batch.print_color(
                    Point::new(mx, my),
                    "g",
                    ColorPair::new(color, RGB::named(BLACK)),
                );
            }
        }

        // Key item — rendered only if not yet picked up
        if let Some(kpos) = self.key_pos {
            if self.map.visible[kpos] {
                let (kx, ky) = xy(kpos);
                draw_batch.print_color(
                    Point::new(kx, ky),
                    "k",
                    ColorPair::new(RGB::from_f32(1.0, 0.85, 0.0), RGB::named(BLACK)),
                );
            }
        }

        // Player
        let (px, py) = xy(self.player_pos);
        draw_batch.print_color(
            Point::new(px, py),
            "@",
            ColorPair::new(RGB::named(YELLOW), RGB::named(BLACK)),
        );

        // HUD row 48: log message (pre-padded to 80 chars)
        draw_batch.print_color(
            Point::new(0, 48),
            self.log.as_str(),
            ColorPair::new(RGB::named(CYAN), RGB::named(BLACK)),
        );

        // HUD row 49: key status indicator
        let key_status = if self.has_key {
            hud("[KEY: acquired] Reach the exit (>)!")
        } else {
            hud("[KEY: not found] Find the key (k) to unlock the exit.")
        };
        draw_batch.print_color(
            Point::new(0, 49),
            key_status.as_str(),
            ColorPair::new(
                if self.has_key { RGB::from_f32(1.0, 0.85, 0.0) } else { RGB::named(GREY) },
                RGB::named(BLACK),
            ),
        );

        draw_batch.submit(0).expect("Batch error");
        render_draw_buffer(ctx).expect("Render error");
    }

    fn render_level_clear(&self, ctx: &mut BTerm) {
        debug_assert!(
            self.level < MAX_LEVEL,
            "LevelClear state entered at final level"
        );
        let mut draw_batch = DrawBatch::new();
        draw_batch.cls();

        draw_batch.print_color(
            Point::new(28, 22),
            format!("  Level {} Cleared!  ", self.level),
            ColorPair::new(RGB::named(GREEN), RGB::named(BLACK)),
        );
        draw_batch.print_color(
            Point::new(21, 24),
            format!("Press Space to continue to Level {}", self.level + 1),
            ColorPair::new(RGB::named(WHITE), RGB::named(BLACK)),
        );

        draw_batch.submit(0).expect("Batch error");
        render_draw_buffer(ctx).expect("Render error");
    }

    fn render_game_clear(&self, ctx: &mut BTerm) {
        let mut draw_batch = DrawBatch::new();
        draw_batch.cls();

        draw_batch.print_color(
            Point::new(19, 21),
            "  All Levels Cleared!  GAME COMPLETE!  ",
            ColorPair::new(RGB::named(YELLOW), RGB::named(BLACK)),
        );
        draw_batch.print_color(
            Point::new(30, 23),
            "Thanks for playing!",
            ColorPair::new(RGB::named(CYAN), RGB::named(BLACK)),
        );
        draw_batch.print_color(
            Point::new(30, 25),
            "Press R to play again",
            ColorPair::new(RGB::named(WHITE), RGB::named(BLACK)),
        );

        draw_batch.submit(0).expect("Batch error");
        render_draw_buffer(ctx).expect("Render error");
    }

    fn render_game_over(&self, ctx: &mut BTerm) {
        let mut draw_batch = DrawBatch::new();
        draw_batch.cls();

        draw_batch.print_color(
            Point::new(23, 22),
            "    A goblin caught you!  GAME OVER   ",
            ColorPair::new(RGB::named(RED), RGB::named(BLACK)),
        );
        draw_batch.print_color(
            Point::new(30, 24),
            "Press R to play again",
            ColorPair::new(RGB::named(WHITE), RGB::named(BLACK)),
        );

        draw_batch.submit(0).expect("Batch error");
        render_draw_buffer(ctx).expect("Render error");
    }
}

// ── Game loop ─────────────────────────────────────────────────────────────────

impl GameState for State {
    fn tick(&mut self, ctx: &mut BTerm) {
        match self.run_state {
            RunState::Running => {
                let mut player_moved = false;
                if let Some(key) = ctx.key {
                    match key {
                        VirtualKeyCode::Up | VirtualKeyCode::W | VirtualKeyCode::Numpad8 => {
                            self.try_move_player(0, -1);
                            player_moved = true;
                        }
                        VirtualKeyCode::Down | VirtualKeyCode::S | VirtualKeyCode::Numpad2 => {
                            self.try_move_player(0, 1);
                            player_moved = true;
                        }
                        VirtualKeyCode::Left | VirtualKeyCode::A | VirtualKeyCode::Numpad4 => {
                            self.try_move_player(-1, 0);
                            player_moved = true;
                        }
                        VirtualKeyCode::Right | VirtualKeyCode::D | VirtualKeyCode::Numpad6 => {
                            self.try_move_player(1, 0);
                            player_moved = true;
                        }
                        _ => {}
                    }
                }

                if player_moved {
                    self.update_fov();
                }

                // Monsters move on their own timer, independent of player input.
                // Speed increases each level: level 1 = 800 ms, level 5 = 400 ms.
                let monster_interval =
                    MONSTER_BASE_MS - (self.level - 1) as f32 * MONSTER_SPEED_STEP_MS;
                self.monster_timer += ctx.frame_time_ms;
                if self.monster_timer >= monster_interval && self.run_state == RunState::Running {
                    self.monster_timer = 0.0;
                    self.move_monsters();
                }

                // Render the appropriate screen immediately on state change
                match self.run_state {
                    RunState::Running => self.render(ctx),
                    RunState::LevelClear => self.render_level_clear(ctx),
                    RunState::GameClear => self.render_game_clear(ctx),
                    RunState::Lost => self.render_game_over(ctx),
                }
            }
            RunState::LevelClear => {
                self.render_level_clear(ctx);
                if matches!(
                    ctx.key,
                    Some(VirtualKeyCode::Space) | Some(VirtualKeyCode::Return)
                ) {
                    let next = self.level + 1;
                    *self = State::build(next);
                }
            }
            RunState::GameClear => {
                self.render_game_clear(ctx);
                if let Some(VirtualKeyCode::R) = ctx.key {
                    *self = State::build(1);
                }
            }
            RunState::Lost => {
                self.render_game_over(ctx);
                if let Some(VirtualKeyCode::R) = ctx.key {
                    *self = State::build(1);
                }
            }
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> BError {
    let context = BTermBuilder::simple80x50()
        .with_title("Dungeon Postman")
        .build()?;
    main_loop(context, State::build(1))
}
