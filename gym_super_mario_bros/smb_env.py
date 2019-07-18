"""An OpenAI Gym environment for Super Mario Bros. and Lost Levels."""
from collections import defaultdict
from nes_py import NESEnv
import numpy as np
from ._roms import decode_target
from ._roms import rom_path
from math import floor, fabs


# create a dictionary mapping value of status register to string names
_STATUS_MAP = defaultdict(lambda: 'fireball', {0:'small', 1: 'tall'})


# a set of state values indicating that Mario is "busy"
_BUSY_STATES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07]


# RAM addresses for enemy types on the screen
_ENEMY_TYPE_ADDRESSES = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A]


# enemies whose context indicate that a stage change will occur (opposed to an
# enemy that implies a stage change wont occur -- i.e., a vine)
# Bowser = 0x2D
# Flagpole = 0x31
_STAGE_OVER_ENEMIES = np.array([0x2D, 0x31])

addr_enemy_page = 0x6e
addr_enemy_x = 0x87
addr_enemy_y = 0xcf
addr_tiles = 0x500
addr_curr_y = 0
checkpoint = 50

class SuperMarioBrosEnv(NESEnv):
    """An environment for playing Super Mario Bros with OpenAI Gym."""

    # the legal range of rewards for each step
    reward_range = (-15, 15)

    def __init__(self, rom_mode='vanilla', lost_levels=False, target=None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            lost_levels (bool): whether to load the ROM with lost levels.
                - False: load original Super Mario Bros.
                - True: load Super Mario Bros. Lost Levels
            target (tuple): a tuple of the (world, stage) to play as a level

        Returns:
            None

        """
        # decode the ROM path based on mode and lost levels flag
        rom = rom_path(lost_levels, rom_mode)
        # initialize the super object with the ROM path
        super(SuperMarioBrosEnv, self).__init__(rom)
        # set the target world, stage, and area variables
        target = decode_target(target, lost_levels)
        self._target_world, self._target_stage, self._target_area = target
        # setup a variable to keep track of the last frames time
        self._time_last = 0
        # setup a variable to keep track of the last frames x position
        self._x_position_last = 0
        self.r = 0
        # reset the emulator
        self.reset()
        # skip the start screen
        self._skip_start_screen()
        # create a backup state to restore from on subsequent calls to reset
        self._backup()

    @property
    def is_single_stage_env(self):
        """Return True if this environment is a stage environment."""
        return self._target_world is not None and self._target_area is not None

    # MARK: Memory access

    def _read_mem_range(self, address, length):
        """
        Read a range of bytes where each byte is a 10's place figure.

        Args:
            address (int): the address to read from as a 16 bit integer
            length: the number of sequential bytes to read

        Note:
            this method is specific to Mario where three GUI values are stored
            in independent memory slots to save processing time
            - score has 6 10's places
            - coins has 2 10's places
            - time has 3 10's places

        Returns:
            the integer value of this 10's place representation

        """
        return int(''.join(map(str, self.ram[address:address + length])))

    @property
    def _level(self):
        """Return the level of the game."""
        return self.ram[0x075f] * 4 + self.ram[0x075c]

    @property
    def _world(self):
        """Return the current world (1 to 8)."""
        return self.ram[0x075f] + 1

    @property
    def _stage(self):
        """Return the current stage (1 to 4)."""
        return self.ram[0x075c] + 1

    @property
    def _area(self):
        """Return the current area number (1 to 5)."""
        return self.ram[0x0760] + 1

    @property
    def _score(self):
        """Return the current player score (0 to 999990)."""
        # score is represented as a figure with 6 10's places
        return self._read_mem_range(0x07de, 6)

    @property
    def _time(self):
        """Return the time left (0 to 999)."""
        # time is represented as a figure with 3 10's places
        return self._read_mem_range(0x07f8, 3)

    @property
    def _coins(self):
        """Return the number of coins collected (0 to 99)."""
        # coins are represented as a figure with 2 10's places
        return self._read_mem_range(0x07ed, 2)

    @property
    def _life(self):
        """Return the number of remaining lives."""
        return self.ram[0x075a]

    @property
    def _x_position(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return self.ram[0x6d] * 0x100 + self.ram[0x86]

    @property
    def _x_position_lol(self):
        """Return the current horizontal position."""
        # add the current page 0x6d to the current x
        return self.ram[0x6d] * 0x100 + self.ram[0x86] + 20

    @property
    def _left_x_position(self):
        """Return the number of pixels from the left of the screen."""
        # subtract the left x position 0x071c from the current x 0x86
        return (self.ram[0x86] - self.ram[0x071c]) % 256

    @property
    def _y_pixel(self):
        """Return the current vertical position."""
        return self.ram[0x03b8]

    @property
    def _y_viewport(self):
        """
        Return the current y viewport.

        Note:
            1 = in visible viewport
            0 = above viewport
            > 1 below viewport (i.e. dead, falling down a hole)
            up to 5 indicates falling into a hole

        """
        return self.ram[0x00b5]

    @property
    def _y_position(self):
        """Return the current vertical position."""
        # check if Mario is above the viewport (the score board area)
        if self._y_viewport < 1:
            # y position overflows so we start from 255 and add the offset
            return 255 + (255 - self._y_pixel)
        # invert the y pixel into the distance from the bottom of the screen
        return 255 - self._y_pixel

    @property
    def _player_status(self):
        """Return the player status as a string."""
        return _STATUS_MAP[self.ram[0x0756]]

    @property
    def _player_state(self):
        """
        Return the current player state.

        Note:
            0x00 : Leftmost of screen
            0x01 : Climbing vine
            0x02 : Entering reversed-L pipe
            0x03 : Going down a pipe
            0x04 : Auto-walk
            0x05 : Auto-walk
            0x06 : Dead
            0x07 : Entering area
            0x08 : Normal
            0x09 : Cannot move
            0x0B : Dying
            0x0C : Palette cycling, can't move

        """
        return self.ram[0x000e]

    @property
    def _is_dying(self):
        """Return True if Mario is in dying animation, False otherwise."""
        return self._player_state == 0x0b or self._y_viewport > 1

    @property
    def _is_dead(self):
        """Return True if Mario is dead, False otherwise."""
        return self._player_state == 0x06

    @property
    def _is_game_over(self):
        """Return True if the game has ended, False otherwise."""
        # the life counter will get set to 255 (0xff) when there are no lives
        # left. It goes 2, 1, 0 for the 3 lives of the game
        return self._life == 0xff

    @property
    def _is_busy(self):
        """Return boolean whether Mario is busy with in-game garbage."""
        return self._player_state in _BUSY_STATES

    @property
    def _is_world_over(self):
        """Return a boolean determining if the world is over."""
        # 0x0770 contains GamePlay mode:
        # 0 => Demo
        # 1 => Standard
        # 2 => End of world
        return self.ram[0x0770] == 2

    @property
    def _is_stage_over(self):
        """Return a boolean determining if the level is over."""
        # check if Bowser of Flagpole enemy types are loaded into RAM to
        # prevent accidental stage change detection when Mario is using a vine
        # (because 0x001D will get set to 3 in this case)
        if np.isin(self.ram[_ENEMY_TYPE_ADDRESSES], _STAGE_OVER_ENEMIES).any():
            # check if the "float state" of the agent is 3
            return self.ram[0x001D] == 3

        return False
    @property
    def _is_enemy(self):
        """Return a boolean determining if the level is over."""
        # check if Bowser of Flagpole enemy types are loaded into RAM to
        # prevent accidental stage change detection when Mario is using a vine
        # (because 0x001D will get set to 3 in this case)
        hols = []
        for i in _ENEMY_TYPE_ADDRESSES:
            hols.append(self.ram[i])
        return hols

    @property
    def _get_enemies(self):
        enemies = []
        for slot in range(0,4):
            local_enemy = self.ram[(0xF + slot)]
            if local_enemy != 0:
                local_ex = self.ram[(addr_enemy_page + slot)] * 0x100 + self.ram[(addr_enemy_x + slot)]
                local_ey = self.ram[(addr_enemy_y + slot)]
                enemies.append({"x" : local_ex,"y" : local_ey})
        return enemies

    @property
    def _flag_get(self):
        """Return a boolean determining if the agent reached a flag."""
        return self._is_world_over or self._is_stage_over

    # MARK: RAM Hacks

    def _write_stage(self):
        """Write the stage data to RAM to overwrite loading the next stage."""
        self.ram[0x075f] = self._target_world - 1
        self.ram[0x075c] = self._target_stage - 1
        self.ram[0x0760] = self._target_area - 1

    def _runout_prelevel_timer(self):
        """Force the pre-level timer to 0 to skip frames during a death."""
        self.ram[0x07A0] = 0

    def _skip_change_area(self):
        """Skip change area animations by by running down timers."""
        change_area_timer = self.ram[0x06DE]
        if change_area_timer > 1 and change_area_timer < 255:
            self.ram[0x06DE] = 1

    def _skip_occupied_states(self):
        """Skip occupied states by running out a timer and skipping frames."""
        while self._is_busy or self._is_world_over:
            self._runout_prelevel_timer()
            self._frame_advance(0)

    def _skip_start_screen(self):
        """Press and release start to skip the start screen."""
        # press and release the start button
        self._frame_advance(8)
        self._frame_advance(0)
        # Press start until the game starts
        while self._time == 0:
            # press and release the start button
            self._frame_advance(8)
            # if we're in the single stage, environment, write the stage data
            # NO BORRAR
            if self.is_single_stage_env:
                self._write_stage()
            self._frame_advance(0)
            # run-out the prelevel timer to skip the animation
            self._runout_prelevel_timer()
        # set the last time to now
        self._time_last = self._time
        # after the start screen idle to skip some extra frames
        while self._time >= self._time_last:
            self._time_last = self._time
            self._frame_advance(8)
            self._frame_advance(0)

    def _skip_end_of_world(self):
        """Skip the cutscene that plays at the end of a world."""
        if self._is_world_over:
            # get the current game time to reference
            time = self._time
            # loop until the time is different
            while self._time == time:
                # frame advance with NOP
                self._frame_advance(0)

    def _kill_mario(self):
        """Skip a death animation by forcing Mario to death."""
        # force Mario's state to dead
        self.ram[0x000e] = 0x06
        # step forward one frame
        self._frame_advance(0)

    # MARK: Reward Function

    @property
    def _x_reward(self):
        """Return the reward based on left right movement between steps."""
        #print(self._x_position)
        _reward = self._x_position - self._x_position_last
        #print("PRIMER REWARD {}".format(_reward))
        self._x_position_last = self._x_position
        r = 0
        #if _reward > 0:
        #    r = 150*_reward
        #elif _reward < 0:
        #    r = -200*abs(_reward)
        #else:
        #    r = -350
        #print("ñe ", r)
        #if _reward < -5 or _reward > 5:
        #    return 0
        if _reward == 0:
            return -10
        else:
            return 0
        return r

    @property
    def _time_penalty(self):
        """Return the reward for the in-game clock ticking."""
        _reward = self._time - self._time_last
        _reward = 0
        #print("taim ", _reward*10)
        if self._time == 0:
            return -5000
        if _reward > 0:
            return 0
            
        return 5

    @property
    def _death_penalty(self):
        """Return the reward earned by dying."""
        if self._is_dying or self._is_dead:
            return -5000

        return 0

    # MARK: nes-py API calls

    def _will_reset(self):
        """Handle and RAM hacking before a reset occurs."""
        self._time_last = 0
        self._x_position_last = 0

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self._time_last = self._time
        self._x_position_last = self._x_position

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        # if done flag is set a reset is incoming anyway, ignore any hacking
        if done:
            return
        # if mario is dying, then cut to the chase and kill hi,
        if self._is_dying:
            self._kill_mario()
        # skip world change scenes (must call before other skip methods)
        #NO BORRAR
        if not self.is_single_stage_env:
            self._skip_end_of_world()
        # skip area change (i.e. enter pipe, flag get, etc.)
        self._skip_change_area()
        # skip occupied states like the black screen between lives that shows
        # how many lives the player has left
        self._skip_occupied_states()

    #@property
    def _get_reward(self):
        """Return the reward after a step occurs."""
        self.r = self._x_reward + self._death_penalty + self._time_penalty
        if self._flag_get:
            self.r += 1000000
        return self.r

    @property
    def _reward(self):
        return self._x_reward + self._time_penalty + self._death_penalty

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        # SI BORRAR SOLO SI SE QUIERE TENER LAS 3 VIDAS EN UN MAPA X
        if self.is_single_stage_env:
            return self._is_dying or self._is_dead or self._flag_get
        return self._is_game_over

    def _get_tile_type(self, box_x, box_y):
        left_x = self._left_x_position
        local_x = self._x_position - left_x + box_x + 112
        local_y = box_y + 96
        local_page = floor(local_x / 256) % 2
        sub_x = floor((local_x % 256) / 16)
        sub_y = floor((local_y - 32) / 16)
        curr_tile_addr = addr_tiles + local_page * 13 * 16 + sub_y * 16 + sub_x
        if (sub_y >= 13) or (sub_y < 0):
            return 0
        # 0 = empty space, 1 is not-empty (e.g. hard surface or object)
        if self.ram[(curr_tile_addr)] != 0:
            return 1
        else:
            return 0

    @property
    def _get_tiles(self):
        enemies = self._get_enemies
        left_x = self._left_x_position
        y_viewport = self._y_viewport
        #Outside box (80 x 65 px)
        #Will contain a matrix of 16x13 sub-boxes of 5x5 pixels each
        box = []
        boxi = []
        y = 0
        z = 0
        for box_y in range(-4*16,8*16,16):
            local_tile_string = "";
            local_data_count = 0;
            for box_x in range(-7*16,8*16,16):
                #0 = Empty space
                tile_value = 0
                #+1 = Not-Empty space (e.g. hard surface, object)
                curr_tile_type = self._get_tile_type(box_x, box_y)
                #tile_value = curr_tile_type
                if (curr_tile_type == 1) and (self._y_position + box_y < 0x1B0):
                    tile_value = 1
               #  +2 = Enemies
                for i in enemies:
                    dist_x = fabs(i["x"] - (self._x_position + box_x - left_x + 108))
                    dist_y = fabs(i["y"] - (90 + box_y))
                    if (dist_x <= 8) and (dist_y <= 8):
                        tile_value = 2
                #+3 = Mario
                dist_x = fabs(self._x_position - (self._x_position + box_x - left_x + 108))
                dist_y = fabs(self._y_pixel - (80 + box_y))
                if (y_viewport == 1) and (dist_x <= 8) and (dist_y <= 8):
                    tile_value = 3
                boxi.append(tile_value)
            box.append(boxi)
            boxi = []
        return box

    def _get_info(self):
        """Return the info after a step occurs"""
        return dict(
            coins=self._coins,
            flag_get=self._flag_get,
            life=self._life,
            score=self._score,
            stage=self._stage,
            status=self._player_status,
            time=self._time,
            world=self._world,
            x_pos=self._x_position,
            y_pos=self._y_position,
            enemy = self._get_tiles,
            reward = self.r,
            #reward = self.reward,
        )


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosEnv.__name__]
