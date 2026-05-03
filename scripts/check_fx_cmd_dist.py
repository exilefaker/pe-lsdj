from pe_lsdj import SongFile                                                                                                                         
from pe_lsdj.constants import FX_COMMANDS

sf = SongFile("data/generated_reg/CHINA.lsdsng")                                                                                                            
fx = sf.song_tokens[:, :, 2]  # (S, 4) phrase fx_cmd column                                                                                          
                                                                                                                                                    
import numpy as np
for cmd_val in range(19):                                                                                                                            
    count = int((fx == cmd_val).sum())                                                                                                               
    if count > 0:                                                                                                                                    
        print(f"CMD {FX_COMMANDS[cmd_val]}: {count:4d} occurrences") 