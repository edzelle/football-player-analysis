using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FootballDataReader.Models.Constants
{
    public class FootballSql
    {
        public const string GetTeamNameForPlayerId = @"select team_name from football.teams t,
                                                football.players p,
                                                football.player_plays_for_team pp 
                                                where p.name = @playerId
                                                and p.id = pp.player_id
                                                and pp.team_id = t.id;";
    }
}
